"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


DROP_LINE_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_IMG_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _is_remove_img_code(code: str) -> bool:
    return code in REMOVE_IMG_CODES or code.startswith("DUP_IMAGE_")


def fix(ds, issues, cfg):
    err_issues = [i for i in issues if i.severity == ERROR]
    fixes = []
    removed_images = set()

    # 1. Xoá ảnh + nhãn bị loại ở cửa vào / trùng / rò rỉ
    for i in err_issues:
        if _is_remove_img_code(i.code):
            stem = i.image
            if stem in removed_images:
                continue
            img_p = ds.image_path(stem)
            lbl_p = ds.label_path(stem)
            had_any = False
            before_desc = stem
            if img_p is not None and img_p.exists():
                before_desc = img_p.name
                img_p.unlink()
                had_any = True
            if lbl_p.exists():
                lbl_p.unlink()
                had_any = True
            ds._size_cache.pop(stem, None)
            removed_images.add(stem)
            if had_any:
                fixes.append(
                    {
                        "image": stem,
                        "line": "",
                        "action": "remove_image",
                        "before": before_desc,
                        "after": "(đã xoá ảnh + nhãn)",
                    }
                )

    # 2. Sửa tên lớp trong data.yaml nếu có lỗi BAD_CLASS_NAME
    if any(i.code == "BAD_CLASS_NAME" for i in err_issues):
        expected = {idx: name for idx, name in enumerate(cfg["classes"])}
        cur_names = ds.yaml_names()
        if cur_names != expected:
            data = {}
            if ds.yaml_path.exists():
                data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            before_names = data.get("names", cur_names)
            data["names"] = expected
            ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            fixes.append(
                {
                    "image": "data.yaml",
                    "line": "",
                    "action": "rename_classes",
                    "before": str(before_names),
                    "after": str(cfg["classes"]),
                }
            )

    # 3. Ghi nhận ảnh thiếu file nhãn (MISSING_LABEL_FILE) -> chuyển người gán, không tạo file rỗng
    seen_missing = set()
    for i in err_issues:
        if i.code == "MISSING_LABEL_FILE":
            stem = i.image
            if stem in removed_images or stem in seen_missing:
                continue
            if not ds.label_path(stem).exists():
                seen_missing.add(stem)
                fixes.append(
                    {
                        "image": stem,
                        "line": "",
                        "action": "needs_label",
                        "before": "(không có file)",
                        "after": "chuyển người gán",
                    }
                )

    # 4. Gom các lỗi theo dòng trong từng file nhãn và sửa 1 lần trên mỗi file
    line_codes_by_stem = defaultdict(lambda: defaultdict(set))
    for i in err_issues:
        if i.code in DROP_LINE_CODES or i.code == "BOX_OUT_OF_IMAGE":
            if i.image in removed_images or i.line is None:
                continue
            line_codes_by_stem[i.image][int(i.line)].add(i.code)

    for stem in sorted(line_codes_by_stem):
        lbl_p = ds.label_path(stem)
        if not lbl_p.exists():
            continue
        raw_lines = lbl_p.read_text(encoding="utf-8", errors="replace").splitlines()
        line_map = line_codes_by_stem[stem]
        new_lines = []
        changed = False

        for idx, orig in enumerate(raw_lines, 1):
            codes = line_map.get(idx)
            if not codes:
                new_lines.append(orig)
                continue

            if codes & DROP_LINE_CODES:
                fixes.append(
                    {
                        "image": stem,
                        "line": idx,
                        "action": "drop_line",
                        "before": orig,
                        "after": "",
                    }
                )
                changed = True
                continue

            if "BOX_OUT_OF_IMAGE" in codes:
                try:
                    cls_id, cx, cy, w, h = parse_yolo_line(orig)
                except ValueError:
                    fixes.append(
                        {
                            "image": stem,
                            "line": idx,
                            "action": "drop_line",
                            "before": orig,
                            "after": "",
                        }
                    )
                    changed = True
                    continue

                nx1 = max(0.0, cx - w / 2.0)
                ny1 = max(0.0, cy - h / 2.0)
                nx2 = min(1.0, cx + w / 2.0)
                ny2 = min(1.0, cy + h / 2.0)
                nw = nx2 - nx1
                nh = ny2 - ny1

                if nw <= 0.0 or nh <= 0.0:
                    fixes.append(
                        {
                            "image": stem,
                            "line": idx,
                            "action": "drop_line",
                            "before": orig,
                            "after": "",
                        }
                    )
                    changed = True
                else:
                    ncx = (nx1 + nx2) / 2.0
                    ncy = (ny1 + ny2) / 2.0
                    clipped = f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"
                    new_lines.append(clipped)
                    fixes.append(
                        {
                            "image": stem,
                            "line": idx,
                            "action": "clip_box",
                            "before": orig,
                            "after": clipped,
                        }
                    )
                    changed = True

        if changed:
            lbl_p.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    return fixes
