# Đánh giá phát triển theo thời gian — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dùng validation theo thời gian cho phát triển mô hình và ngừng chấm test ngẫu nhiên cũ đã bị expose.

**Architecture:** Giữ nguyên thứ tự hàng do UCI cung cấp. Training và comparison dùng 80% đầu làm train, 20% cuối làm validation; không tạo test split hay tính test metrics. Tài liệu nêu rõ đây là validation phục vụ phát triển, không phải ước lượng độc lập.

**Tech Stack:** Python, pandas, scikit-learn, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-29-temporal-development-evaluation-design.md`

## Global Constraints

- `bank-full.csv` được nguồn UCI mô tả là sắp theo ngày từ tháng 5/2008 đến tháng 11/2010.
- Chia theo đúng thứ tự hàng: 80% đầu là training, 20% cuối là validation.
- Không shuffle hoặc stratify vì sẽ phá thứ tự thời gian.
- Không tạo hoặc công bố holdout độc lập mới từ `bank-full.csv` hiện có.
- Không chọn/tối ưu threshold hoặc hyperparameter trong thay đổi này.
- Không train lại/ghi đè model artifacts như một phần của thay đổi đánh giá.
- Không tạo timestamp giả từ `day`/`month`, vì dataset không có năm/ngày đầy đủ cho từng row.

## Review Focus

- Nhãn và feature giữ nguyên thứ tự thời gian sau split; test kiểm tra các marker row ở hai phía ranh giới.
- Dataset có ít hơn hai hàng hoặc thiếu hàng: không phát sinh split train/validation rỗng nếu có thể xác thực trước khi train; test xác nhận cách xử lý input nhỏ.
- Training không gọi `predict_proba`/evaluator trên test cũ hoặc in test metrics; test kiểm tra orchestration chỉ chấm validation.
- Comparison không nhận/truy cập test split; test kiểm tra kết quả chỉ có validation.
- Các metric validation không bị mô tả là holdout độc lập; README/PLAN phải nói rõ cần dữ liệu chiến dịch mới.

---

### Task 1: Chuyển pipeline sang train/validation theo thứ tự thời gian

**Files:**
- Modify: `scripts/train_model.py`
- Modify: `scripts/compare_models.py`
- Modify: `tests/test_feature_contract.py`
- Modify: `tests/test_compare_models.py`
- Test: `tests/test_train_model.py`

**Interfaces:**
- `scripts.train_model.load_and_split_data()` trả `(X_train, X_val, y_train, y_val)`.
- `scripts.compare_models.load_and_split_data()` trả `(X_train, X_val, y_train, y_val)`.
- Cả hai dùng cùng cách cắt chỉ số `split_index = int(len(data) * 0.8)` và giữ thứ tự hàng.

- [x] **Step 1: Viết test đỏ cho split theo thời gian**
  - Tạo `tests/test_train_model.py` với fake dataset 10 hàng có feature marker tăng dần và target xen kẽ.
  - Xác nhận 8 hàng đầu ở train, 2 hàng cuối ở validation; marker feature và target cùng giữ thứ tự.
  - Cập nhật `tests/test_feature_contract.py` để kiểm tra contract 11 cột và thứ tự hàng qua split 80/20.
  - Cập nhật `tests/test_compare_models.py`: 1000 hàng phải cho 800 train/200 validation; xác nhận không split ngẫu nhiên và không có test return value.

- [x] **Step 2: Chạy tests để xác nhận thất bại đúng**

Run: `uv run pytest tests/test_train_model.py tests/test_feature_contract.py tests/test_compare_models.py -q`

Expected: FAIL vì split hiện tại đang stratified/random và API trả test split.

- [x] **Step 3: Đổi hai loader sang split theo thứ tự hàng**
  - Chọn `PRE_CAMPAIGN_FEATURES`, encode target, rồi cắt features/target tại `int(len(df) * 0.8)` mà không shuffle.
  - Trả bốn giá trị train/validation theo Interfaces.
  - Từ chối dữ liệu có ít hơn hai hàng bằng `ValueError` trước khi chia.

- [x] **Step 4: Bỏ test scoring khỏi training/comparison**
  - Trong `scripts/train_model.py`, cập nhật caller theo API bốn giá trị; chỉ đánh giá validation, lưu pipeline/metadata như hiện tại, và xóa phần predict/evaluate/in kết quả test.
  - Trong `scripts/compare_models.py`, cập nhật caller theo API bốn giá trị và chỉ ghi kết quả validation.
  - Thêm test orchestration xác nhận training chỉ đánh giá validation và không gọi evaluator/predict trên test.

- [x] **Step 5: Chạy tests mục tiêu**

Run: `uv run pytest tests/test_train_model.py tests/test_feature_contract.py tests/test_compare_models.py -q`

Expected: PASS.

### Task 2: Công bố giới hạn đánh giá và hướng dẫn đánh giá tiếp theo

**Files:**
- Modify: `README.md`
- Modify: `PLAN.md`
- Modify: `docs/superpowers/specs/2026-10-29-temporal-development-evaluation-design.md`

- [x] **Step 1: Cập nhật README**
  - Mô tả split train 80% đầu / validation 20% cuối theo thứ tự UCI.
  - Ghi rõ không có test metrics/holdout độc lập từ dataset hiện tại vì test cũ đã bị expose.
  - Nêu đánh giá cuối cần dữ liệu campaign mới chưa tham gia model/threshold selection.
  - Sửa hướng dẫn training để không còn nói script in test metrics.

- [x] **Step 2: Cập nhật PLAN.md**
  - Đánh dấu các mục validation-only/không dùng test cũ và temporal development split hoàn tất.
  - Giữ mục final independent evaluation chưa hoàn tất cho tới khi có campaign data mới.
  - Ghi số tests mới nhất sau verification.

- [x] **Step 3: Kiểm tra tài liệu**

Run: `rg -n "test metrics|test set|holdout|validation" README.md`

Expected: README không tuyên bố test metrics hiện tại là unbiased; mọi holdout độc lập được gắn điều kiện có dữ liệu mới.

### Task 3: Xác minh toàn bộ thay đổi

**Files:** Không thêm file.

- [x] **Step 1: Chạy toàn bộ test suite**

Run: `uv run pytest -v`

Expected: toàn bộ tests pass; không cần dữ liệu ngoài cho unit tests.

- [x] **Step 2: Rà diff**

Run: `git diff --check`

Expected: không có whitespace errors và không có thay đổi trong `models/`.
