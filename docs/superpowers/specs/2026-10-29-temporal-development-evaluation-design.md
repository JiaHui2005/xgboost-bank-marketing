# Đánh giá phát triển theo thời gian

## Mục đích

Không tiếp tục chấm điểm trên test split ngẫu nhiên cũ đã bị expose; thay bằng validation theo thứ tự thời gian để phát triển mô hình, và không tuyên bố có ước lượng holdout độc lập từ dataset hiện tại.

## Quyết định và hướng thực hiện

Tài liệu UCI đi kèm xác nhận `bank-full.csv` được sắp theo ngày từ tháng 5/2008 đến tháng 11/2010. Dùng nguyên thứ tự hàng: 80% đầu làm training, 20% cuối làm validation. Không shuffle hoặc stratify vì sẽ phá thứ tự thời gian. Tỷ lệ 80/20 là split phát triển, không phải holdout độc lập.

Training và comparison chỉ báo cáo validation metrics; bỏ bước predict/evaluate trên test cũ. Validation được dùng cho phát triển/chọn model, không phải ước lượng unbiased. Hiện chưa có đánh giá độc lập; đánh giá cuối cần dữ liệu chiến dịch mới chưa từng được dùng để chọn mô hình, hyperparameter hoặc ngưỡng.

## Phạm vi

- Đồng bộ split theo thứ tự thời gian trong training và model comparison.
- Bỏ test split/scoring khỏi luồng training và đảm bảo comparison không truy cập test.
- Cập nhật tests để khóa thứ tự train trước validation và xác nhận test cũ không được chấm.
- Cập nhật README và `PLAN.md` để mô tả rõ validation-only, giới hạn do test cũ bị expose, và điều kiện cần có dữ liệu mới cho đánh giá độc lập.

## Ngoài phạm vi và giới hạn

- Không tạo hoặc công bố holdout độc lập mới từ `bank-full.csv` hiện có.
- Không chọn/tối ưu threshold hoặc hyperparameter trong thay đổi này.
- Không train lại/ghi đè model artifacts như một phần của thay đổi đánh giá.
- Thứ tự hàng được dùng theo data dictionary UCI; không tạo timestamp giả từ `day`/`month`, vì dataset không có năm/ngày đầy đủ cho từng row.

## Tiêu chí hoàn thành

Train và validation giữ thứ tự thời gian (80/20), không có code path nào chấm test cũ, tests bảo vệ hành vi validation-only, và tài liệu không mô tả validation như holdout độc lập.
