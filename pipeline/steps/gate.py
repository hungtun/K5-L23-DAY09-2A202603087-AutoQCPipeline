"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN  # noqa: F401


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    stems = set(image_stems)
    reviewed_set = set(reviewed or ())
    n_images = len(stems)
    hold_thr = cfg["gate"]["warn_ratio_hold"]

    errors = [
        i for i in issues
        if i.severity == ERROR and (not stems or i.image in stems or i.image in ("data.yaml", "<dataset>"))
    ]
    n_errors = len(errors)

    warn_images = {
        i.image for i in issues
        if i.severity == WARN and i.image in stems and i.image not in reviewed_set
    }
    n_warn_images = len(warn_images)
    raw_ratio = (n_warn_images / n_images) if n_images > 0 else 0.0
    warn_ratio = round(raw_ratio, 4)

    reasons = []
    if n_errors > 0:
        decision = "Rework"
        reasons.append(f"Còn {n_errors} lỗi bắt buộc chưa được xử lý (cần Rework).")
        if n_warn_images > 0:
            reasons.append(f"Có {n_warn_images}/{n_images} ảnh ({warn_ratio:.2%}) còn cảnh báo chưa duyệt.")
    elif raw_ratio >= hold_thr:
        decision = "Hold"
        reasons.append(
            f"Không còn lỗi bắt buộc, nhưng tỉ lệ ảnh còn cảnh báo chưa duyệt là {n_warn_images}/{n_images} "
            f"({warn_ratio:.2%}) >= ngưỡng Hold ({hold_thr:.0%})."
        )
    else:
        decision = "Release"
        reasons.append(
            f"Không còn lỗi bắt buộc; tỉ lệ ảnh còn cảnh báo chưa duyệt là {n_warn_images}/{n_images} "
            f"({warn_ratio:.2%}) < ngưỡng Hold ({hold_thr:.0%})."
        )

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "reasons": reasons,
    }
