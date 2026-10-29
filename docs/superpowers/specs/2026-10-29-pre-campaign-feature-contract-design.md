# Hợp đồng feature trước chiến dịch

## Mục đích

Xác định đầu vào của mô hình khi dự đoán khách hàng đăng ký trước khi chiến dịch marketing bắt đầu; đồng bộ hợp đồng này giữa huấn luyện, so sánh mô hình, dự đoán hàng loạt, notebook EDA và tài liệu.

## Quyết định và hướng thực hiện

Chỉ sử dụng các feature được xem là có sẵn trước chiến dịch hiện tại. Loại `duration` và `campaign` như dự định ban đầu; đồng thời loại `contact`, `day`, `month` vì bộ dữ liệu mô tả chúng là thông tin của lần liên hệ hiện tại. Giữ `pdays`, `previous`, `poutcome` làm feature lịch sử với giả định chúng chỉ phản ánh hoạt động trước chiến dịch đang được dự đoán.

Hướng thay thế là giữ `contact`, `day` hoặc `month` nếu chúng đại diện cho kế hoạch chiến dịch đã được duyệt trước. Tuy nhiên, bộ dữ liệu hiện tại không hỗ trợ cách hiểu này vì nó ghi nhận lần liên hệ hiện tại. Có thể xem xét lại hợp đồng nếu sau này có nguồn dữ liệu phân biệt giá trị dự kiến với giá trị quan sát thực tế.

## Phạm vi

- Định nghĩa một danh sách feature pre-campaign dùng chung cho huấn luyện, chuẩn bị dữ liệu validation/test, so sánh mô hình và dự đoán hàng loạt.
- Chỉ yêu cầu các feature trong hợp đồng khi inference; bỏ qua các cột nguồn đã loại nếu chúng vẫn xuất hiện trong dữ liệu đầu vào.
- Từ chối model artifact nếu metadata không xác nhận đã loại đủ cả năm feature ngoài hợp đồng; không tự tái huấn luyện/ghi đè artifact.
- Bổ sung regression test xác nhận các trường thuộc chiến dịch hiện tại đã loại không được sử dụng và hợp đồng feature được áp dụng nhất quán.
- Cập nhật README, `PLAN.md` và `notebooks/01_eda.ipynb` để nêu thời điểm dự đoán, lý do chọn feature và giả định có điều kiện về các trường lịch sử.

## Ngoài phạm vi và giới hạn

- Không tạo holdout theo thời gian mới hoặc thay đổi việc lựa chọn mô hình/ngưỡng trong task này.
- Không khẳng định thời điểm của các trường lịch sử đã được xác minh bằng data dictionary vận hành; cần ghi rõ đây là giả định phải được xác nhận.
- Không huấn luyện lại hoặc ghi đè model artifact đang có trong repo như một phần của thay đổi hợp đồng feature.

## Tiêu chí hoàn thành

Huấn luyện, so sánh và inference dùng cùng một tập feature pre-campaign; test đảm bảo `duration`, `campaign`, `contact`, `day`, `month` không thể lọt vào đầu vào mô hình; tài liệu mô tả chính xác thời điểm dự đoán và giả định còn cần xác nhận về các feature lịch sử.
