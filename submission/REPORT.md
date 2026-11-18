# Báo cáo Day 09 — Phạm Hữu Hưng, 2A202603087

## 1. Kết luận trước
Pipeline kết luận **Hold** với mã thoát 10. Lô dữ liệu đầu vào gồm 224 ảnh, bước nhận dữ liệu loại 22 ảnh lỗi nặng còn 202 ảnh đưa vào kiểm tra và xuất ra bản nháp tại `submission/export/clean_v1`. Toàn bộ lỗi bắt buộc đã được xử lý về 0/202 ảnh, nhưng sau khi người duyệt hết ngân sách 80/125 ảnh trong hàng đợi thì vẫn còn 38/202 ảnh chiếm 18.81% có cảnh báo chưa được duyệt, vượt ngưỡng Hold 10%.

## 2. Lỗi và phân việc

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| `CORRUPT_IMAGE` | 2/224 ảnh | Máy | File ảnh hỏng không giải mã được pixel, loại ngay ở cửa vào |
| `BAD_SIZE` | 3/224 ảnh | Máy | Cạnh ngắn dưới 250px, không đạt chuẩn kích thước tối thiểu |
| `DUP_IMAGE_EXACT` | 8/224 ảnh | Máy | Trùng hoàn toàn mã băm MD5 với ảnh đã nhận trước đó |
| `DUP_IMAGE_NEAR` | 4/224 ảnh | Máy | Trùng gần đúng với khoảng cách pHash từ 10 trở xuống |
| `LEAK_TEST` | 5/224 ảnh | Máy | Trùng với ảnh trong tập test hoặc gold_val, loại bỏ để chống rò rỉ |
| `BAD_CLASS_NAME` | 2 lỗi / `data.yaml` | Máy | Sai tên lớp trong `data.yaml`, máy tự ghi lại đúng danh sách chuẩn |
| `BAD_FORMAT` | 3 dòng / 202 ảnh | Máy | Dòng nhãn sai định dạng 5 số YOLO, máy xóa dòng lỗi |
| `BAD_CLASS_ID` | 6 dòng / 202 ảnh | Máy | ID lớp ngoài phạm vi 0..3, máy xóa dòng lỗi |
| `BOX_ZERO` | 4 dòng / 202 ảnh | Máy | Chiều rộng hoặc chiều cao bằng 0, máy xóa dòng lỗi |
| `DUP_BOX` | 7 dòng / 202 ảnh | Máy | Box cùng lớp trùng lặp IoU trên 0.9 trong cùng ảnh, máy xóa dòng lặp |
| `BOX_OUT_OF_IMAGE` | 8 dòng / 202 ảnh | Máy | Box tràn ngoài biên quá 0.002, máy cắt gọn toạ độ về đoạn 0..1 |
| `MISSING_LABEL_FILE` | 6/202 ảnh | Người | Ảnh thiếu file nhãn, máy không được tự bịa nhãn mà chuyển người gán |
| `BOX_HUGE` | 3 dòng / 202 ảnh | Người | Box chiếm trên 95% diện tích ảnh, cần người kiểm tra thực tế |
| `BOX_TINY` | 6 dòng / 202 ảnh | Người | Cạnh ngắn dưới 12px, cần người xác nhận vật thể thật hay nhiễu |
| `CLASS_RARE` | 4 cảnh báo dataset | Người | Các mã lớp lạ xuất hiện dưới 5 lần trước khi autofix xóa dòng hỏng |
| `EMPTY_LABEL` | 12/202 ảnh | Người | File nhãn rỗng, cần người phân biệt ảnh nền thật hay gán sót |
| `POSSIBLE_WRONG_CLASS` | 18 dòng / 202 ảnh | Người | Model lớn dự đoán lớp khác nhãn hiện tại, cần người kiểm tra ngữ nghĩa |
| `POSSIBLE_MISSING` | 53 dòng / 202 ảnh | Người | Model lớn thấy vật nhưng nhãn không có box, cần người bổ sung |
| `POSSIBLE_LOOSE_BOX` | 28 dòng / 202 ảnh | Người | Box cùng lớp nhưng IoU dưới 0.5 so với model lớn, cần người chỉnh lại |
| `POSSIBLE_FALSE_POS` | 25 dòng / 202 ảnh | Người | Box nhãn không khớp dự đoán nào của model lớn, cần người rà soát |

File `05_review_queue.csv` xếp thứ tự bằng cách cộng trọng số từng loại lỗi nhân với độ tự tin của model: các ảnh chưa có nhãn `NEEDS_LABEL` mang trọng số 10 và các ảnh tích tụ nhiều lỗi nặng như `img_0020` đạt điểm 11.524 được đẩy lên đầu hàng đợi. Với ngân sách 80/125 ảnh, người duyệt xử lý trọn vẹn 6/6 ảnh thiếu file nhãn và giải quyết được 94/123 cảnh báo ngữ nghĩa, gồm 18/18 lỗi `POSSIBLE_WRONG_CLASS`, 51/53 lỗi `POSSIBLE_MISSING`, 13/24 lỗi `POSSIBLE_FALSE_POS` và 12/28 lỗi `POSSIBLE_LOOSE_BOX`.

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 10 | `DUP_IMAGE_NEAR`: 1.00 / 0.50 → 1.00 / 0.67 | Giữ Precision tuyệt đối 1.00 để không xóa nhầm ảnh tốt, tăng Recall từ 3/6 lên 4/6 trên tập calib và bắt đủ ảnh thu nhỏ 85% trên tập raw |
| `second_opinion.conf` | 0.25 | 0.6 | `WRONG_CLASS`: 0.56 / 0.83 → 0.62 / 0.83; `MISSING`: 0.50 / 1.00 → 0.57 / 1.00 | Tăng Precision để giảm báo động giả trong hàng đợi từ 143 xuống 125 ảnh mà vẫn giữ tối đa Recall cho lỗi thiếu nhãn và nhầm lớp |

Điểm đánh lừa của calibration nằm ở ngưỡng `duplicates.phash_dist = 16`: trên bảng sweep của tập `calib`, mức 16 cho F1 cao nhất là 0.91 với Precision hiển thị đạt 1.00 và Recall đạt 0.83. Tuy nhiên bảng sweep chỉ tính trên mã `DUP_IMAGE_NEAR` và tập `calib` không có đáp án cho `LEAK_TEST`, nên việc ảnh sạch `calib_027` bị bắt nhầm thành `LEAK_TEST` với `test_027.jpg` không hiện lên thành FP. Khi áp `phash_dist = 16` vào tập `raw` 224 ảnh, số ảnh bị xóa do `LEAK_TEST` nhảy từ 5 lên 11 ảnh và `DUP_IMAGE_NEAR` tăng từ 4 lên 7 ảnh, xóa oan nhiều cặp ảnh hoàn toàn khác nhau như `img_0011` và `img_0008`.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
(a) Chứng minh cửa chặn hoạt động đúng: khi chạy lệnh `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`, ngân sách duyệt bằng 0 nên 6 ảnh `MISSING_LABEL_FILE` không được người gán nhãn. Ở vòng kiểm tra cuối, `gate` phát hiện còn 6 lỗi bắt buộc nên trả về kết luận **Rework**, thoát với mã 20 và chặn hoàn toàn bước xuất dữ liệu khiến thư mục `work/exp_nb/clean_v1` không được tạo ra.

(b) Hai điểm mù của pipeline kèm bằng chứng:
- Ảnh trùng bị cắt cúp và biến đổi mạnh như `calib_038` trong tập calib có khoảng cách pHash lên tới 18, vượt quá ngưỡng an toàn `phash_dist = 10` nên pipeline không bắt được mà không gây xóa nhầm hàng loạt ảnh khác.
- Các lỗi gán nhầm lớp hoặc box vẽ lỏng ở mức ranh giới mà chính model `yolo26l` cũng dự đoán độ tự tin thấp hoặc có IoU xấp xỉ ngưỡng 0.5: trên tập `calib`, 1/6 lỗi `WRONG_CLASS` và 4/5 lỗi `LOOSE_BOX` bị bỏ sót ở ngưỡng `conf = 0.6`.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa chạy. Hiện tại hoàn thiện và kiểm tra toàn bộ phần lõi của pipeline với `make check-core`, phần huấn luyện trên Google Colab sẽ được thực hiện bổ sung sau.

## 6. Quyết định và ghi chú AI
(a) Quyết định cuối cùng cho lô dữ liệu là **Hold**. Để chuyển từ Hold sang **Release**, cần cấp thêm ngân sách người duyệt cho tối thiểu 18 ảnh nữa trong số 38 ảnh còn cảnh báo chưa duyệt, nhằm đưa số ảnh cảnh báo tồn đọng xuống tối đa 20/202 ảnh tức dưới ngưỡng 10%.

(b) Ghi chú dùng AI:
- Prompt chính: Viết 4 module `label_format`, `gate`, `autofix`, `second_opinion` bám sát docstring, đọc mọi ngưỡng từ `cfg`, giữ nguyên số dòng gốc khi có dòng trống và đảm bảo tính idempotent khi chạy lại.
- Cách kiểm code AI: Chạy toàn bộ 79 test bằng `pytest`, chạy `make calibrate` kiểm tra độ nhạy trên tập `calib`, và chạy `make run` đối chiếu trực tiếp các báo cáo từ `01_intake` đến `08_summary.json`.
- Một lần AI sai: Ở bước cắt box tràn biên trong `autofix`, nếu box nằm hoàn toàn ngoài ảnh thì sau khi cắt về đoạn 0..1 chiều rộng hoặc chiều cao sẽ bằng 0 hoặc âm; đã phát hiện và sửa thành xóa dòng `drop_line` thay vì ghi ra box rỗng.
