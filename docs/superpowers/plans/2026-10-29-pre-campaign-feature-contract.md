# Hợp đồng feature trước chiến dịch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dùng thống nhất feature contract 11 cột cho dự đoán trước khi chiến dịch bắt đầu và mô tả trung thực quyết định này trong tài liệu/EDA.

**Architecture:** Khai báo danh sách feature pre-campaign duy nhất trong `src/feature_contract.py`. Training, so sánh mô hình và inference cùng dùng danh sách này; các cột ngoài hợp đồng không được đưa vào model input. Regression tests khóa danh sách và cách sử dụng.

**Tech Stack:** Python, pandas, pytest, Jupyter Notebook (ipynb JSON).

**Spec:** `docs/superpowers/specs/2026-10-29-pre-campaign-feature-contract-design.md`

## Global Constraints

- Prediction point là trước khi chiến dịch marketing hiện tại bắt đầu.
- Loại `duration`, `campaign`, `contact`, `day`, `month` khỏi model features.
- Giữ `pdays`, `previous`, `poutcome` với giả định chúng chỉ phản ánh hoạt động có trước chiến dịch; tài liệu phải nêu đây là giả định cần xác nhận.
- Không tạo holdout mới, đổi model/threshold, hoặc train lại/ghi đè model artifacts trong phạm vi này.
- Inference chỉ yêu cầu các feature trong hợp đồng và bỏ qua cột bị loại nếu cột đó có mặt trong input.

## Review Focus

- Input có chứa cột bị loại (`duration`, `campaign`, `contact`, `day`, `month`): phải không đi vào model input; test ở Task 1.
- Input inference thiếu một trong 11 feature bắt buộc: phải báo lỗi thiếu cột; test ở Task 1.
- Thứ tự cột đầu vào khác thứ tự hợp đồng: chọn feature theo danh sách hợp đồng, không theo thứ tự CSV; test ở Task 1.
- Dữ liệu EDA vẫn giữ cột bị loại để phân tích mô tả: chỉ loại khỏi model features, không xóa khỏi dataframe notebook; kiểm tra trong Task 2.
- `pdays`, `previous`, `poutcome` chưa được xác nhận từ data dictionary vận hành: ghi giả định có điều kiện, không tuyên bố đã xác minh; kiểm tra trong Task 2.

---

### Task 1: Đồng bộ hợp đồng feature trong pipeline

**Files:**
- Create: `src/feature_contract.py`
- Modify: `scripts/train_model.py`
- Modify: `scripts/compare_models.py`
- Modify: `scripts/predict.py`
- Test: `tests/test_feature_contract.py`
- Test: `tests/test_predict.py`
- Test: `tests/test_compare_models.py`

**Interfaces:**
- Produces: `PRE_CAMPAIGN_FEATURES`, một danh sách có thứ tự gồm 11 feature: `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `pdays`, `previous`, `poutcome`.
- Inference giữ tên `REQUIRED_COLUMNS` tương thích hiện tại, nhưng giá trị lấy từ `PRE_CAMPAIGN_FEATURES`.

- [ ] **Step 1: Viết regression tests thất bại trước**
  - Trong `tests/test_feature_contract.py`, xác nhận danh sách bằng đúng 11 cột theo thứ tự ở trên và không có `duration`, `campaign`, `contact`, `day`, `month`.
  - Kiểm thử `scripts.train_model.load_and_split_data()` và `scripts.compare_models.load_and_split_data()` chỉ trả các cột trong hợp đồng, dù dữ liệu nguồn có thêm các cột bị loại.
  - Trong `tests/test_predict.py`, xác nhận input có đủ 11 cột không cần 5 cột bị loại; nếu 5 cột đó có mặt thì chúng không làm thay đổi danh sách feature gửi vào pipeline.
  - Sửa fixture `tests/test_compare_models.py` để khẳng định các cột bị loại đều không có trong tập model input.

- [ ] **Step 2: Chạy test để xác nhận thất bại đúng**

Run: `uv run pytest tests/test_feature_contract.py tests/test_predict.py tests/test_compare_models.py -q`

Expected: FAIL vì feature contract chưa được định nghĩa/chưa được dùng nhất quán.

- [ ] **Step 3: Thêm và áp dụng `PRE_CAMPAIGN_FEATURES`**
  - Tạo `src/feature_contract.py` với danh sách theo đúng thứ tự đã chốt.
  - Trong train và comparison, chọn `X = df[PRE_CAMPAIGN_FEATURES].copy()` trước khi split; không truyền các cột bị loại cho pipeline.
  - Trong inference, đặt `REQUIRED_COLUMNS` theo danh sách dùng chung; chỉ truyền `df[REQUIRED_COLUMNS]` vào pipeline, giữ nguyên các cột input khác trong output.
  - Ghi đủ 5 cột bị loại vào metadata `dropped_features` được sinh bởi training script; không chạy training trong task này.

- [ ] **Step 4: Chạy lại test mục tiêu**

Run: `uv run pytest tests/test_feature_contract.py tests/test_predict.py tests/test_compare_models.py -q`

Expected: PASS.

### Task 2: Đồng bộ quyết định trong tài liệu và notebook EDA

**Files:**
- Modify: `README.md`
- Modify: `PLAN.md`
- Modify: `notebooks/01_eda.ipynb`
- Modify: `docs/superpowers/specs/2026-10-29-pre-campaign-feature-contract-design.md`

**Interfaces:**
- Tài liệu công bố đúng danh sách 11 feature và giải thích lý do loại 5 trường.
- Notebook tiếp tục dùng các cột bị loại cho EDA mô tả nếu cần, nhưng kết luận rõ chúng không thuộc model feature set pre-campaign.

- [ ] **Step 1: Cập nhật README**
  - Thay yêu cầu inference “14 model features” bằng 11 cột hợp đồng.
  - Ghi `contact`, `day`, `month` là thuộc lần liên hệ hiện tại và bị loại cho dự đoán trước chiến dịch.
  - Nêu `pdays`, `previous`, `poutcome` chỉ được dùng với giả định dữ liệu vận hành chứng minh chúng có trước chiến dịch.

- [ ] **Step 2: Cập nhật PLAN.md**
  - Đánh dấu các mục đã hoàn thành về prediction point, rà soát ba feature và đồng bộ feature contract.
  - Giữ mục xác minh lịch sử `pdays`, `previous`, `poutcome` là việc cần xác nhận, không đánh dấu hoàn tất.

- [ ] **Step 3: Cập nhật notebook EDA**
  - Trong markdown phần `day`, `month`, `contact`, đổi kết luận từ “chỉ giữ nếu xác nhận” thành quyết định loại cả ba khỏi model features pre-campaign.
  - Giải thích notebook vẫn phân tích các cột này để mô tả dữ liệu; việc chúng xuất hiện trong EDA không có nghĩa là model được dùng chúng.
  - Thêm/cập nhật phần tóm tắt feature contract 11 cột và giả định về các biến lịch sử.
  - Không xóa dữ liệu, biểu đồ hoặc outputs hiện có khỏi notebook.

- [ ] **Step 4: Kiểm tra nhất quán tài liệu và cấu trúc notebook**

Run: `uv run python -m json.tool notebooks/01_eda.ipynb > /dev/null`

Expected: exit code 0; tìm trong README/notebook không còn nội dung cho rằng `contact`, `day`, `month` có thể mặc định được giữ trong model.

### Task 3: Xác minh toàn bộ thay đổi

**Files:** Không thêm file.

- [ ] **Step 1: Chạy toàn bộ test suite**

Run: `uv run pytest -v`

Expected: toàn bộ tests pass; nếu có test integration phụ thuộc artifact cũ, báo rõ lỗi và không train/ghi đè artifact để che lỗi đó.

- [ ] **Step 2: Rà diff cuối**

Run: `git diff --check`

Expected: không có whitespace errors; xác nhận không có thay đổi trong `models/`.
