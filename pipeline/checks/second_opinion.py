"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    c = cfg["second_opinion"]
    conf_thr = c["conf"]
    min_area_px = c["min_area_px"]
    fp_conf = c["fp_conf"]
    iou_missing = c["iou_missing"]
    iou_match = c["iou_match"]
    iou_loose = c["iou_loose"]
    n_classes = len(ds.classes)

    issues = []
    for img_p in ds.image_files():
        stem = img_p.stem
        if stem not in preds or not ds.label_path(stem).exists():
            continue
        size = ds.image_size(stem)
        if size is None:
            continue
        img_w, img_h = size

        valid_lines = []
        for line_no, text in ds.read_label_lines(stem) or []:
            try:
                b = parse_yolo_line(text)
            except ValueError:
                continue
            if 0 <= b[0] < n_classes and b[3] > 0 and b[4] > 0:
                valid_lines.append((line_no, b))

        label_boxes = [b for _, b in valid_lines]
        raw_preds = preds.get(stem) or []

        # 1. Duyệt từng dự đoán đủ điều kiện (conf >= conf_thr và diện tích px >= min_area_px)
        for r in raw_preds:
            p_cls, p_conf, pcx, pcy, pw, ph = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
            if p_conf < conf_thr:
                continue
            if (pw * img_w) * (ph * img_h) < min_area_px:
                continue

            p_box = (pcx, pcy, pw, ph)
            cands = []
            for l_cls, lcx, lcy, lw, lh in label_boxes:
                iou = iou_xywh(p_box, (lcx, lcy, lw, lh))
                if iou >= iou_missing:
                    cands.append((iou, l_cls))

            if not cands:
                issues.append(
                    Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_MISSING",
                        stem,
                        f"model thấy lớp {p_cls} (conf={p_conf:.2f}) nhưng không có box nhãn khớp",
                        line=None,
                        extra={"conf": round(p_conf, 4), "pred_class": p_cls},
                    )
                )
                continue

            same_cls_ious = [iou for iou, l_cls in cands if l_cls == p_cls]
            if same_cls_ious:
                best_same_iou = max(same_cls_ious)
                if best_same_iou < iou_loose:
                    issues.append(
                        Issue(
                            "second_opinion",
                            WARN,
                            "POSSIBLE_LOOSE_BOX",
                            stem,
                            f"box nhãn lớp {p_cls} lệch/lỏng so với model (IoU={best_same_iou:.2f} < {iou_loose})",
                            line=None,
                            extra={"conf": round(p_conf, 4), "iou": round(best_same_iou, 4)},
                        )
                    )
            else:
                best_other_iou, best_other_cls = max(cands, key=lambda x: x[0])
                if best_other_iou >= iou_match:
                    issues.append(
                        Issue(
                            "second_opinion",
                            WARN,
                            "POSSIBLE_WRONG_CLASS",
                            stem,
                            f"nhãn gán lớp {best_other_cls} nhưng model dự đoán lớp {p_cls} (conf={p_conf:.2f}, IoU={best_other_iou:.2f})",
                            line=None,
                            extra={
                                "conf": round(p_conf, 4),
                                "iou": round(best_other_iou, 4),
                                "pred_class": p_cls,
                                "label_class": best_other_cls,
                            },
                        )
                    )

        # 2. Duyệt từng dòng nhãn hợp lệ để tìm POSSIBLE_FALSE_POS (so với dự đoán có conf >= fp_conf, không lọc diện tích)
        fp_pred_boxes = [
            (float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            for r in raw_preds
            if float(r[1]) >= fp_conf
        ]
        for line_no, (l_cls, lcx, lcy, lw, lh) in valid_lines:
            l_box = (lcx, lcy, lw, lh)
            max_iou = max((iou_xywh(l_box, p_box) for p_box in fp_pred_boxes), default=0.0)
            if max_iou < iou_missing:
                issues.append(
                    Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_FALSE_POS",
                        stem,
                        f"box nhãn lớp {l_cls} không khớp dự đoán nào của model (max IoU={max_iou:.2f})",
                        line=line_no,
                        extra={"iou": round(max_iou, 4)},
                    )
                )

    return issues
