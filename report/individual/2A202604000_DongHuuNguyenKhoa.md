# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202604000**
- **Họ và tên: Đồng Hữu Nguyên Khoa**
- **SEED** (in ở ô đầu notebook): `9538`

## 1. Bằng chứng

| Tệp                                                                       | Có? |
| -------------------------------------------------------------------------- | ---- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | ☐   |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐   |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước        | ☐   |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*


                                 precision  recall  F1 (test)  F1 (SUB 40)  \

1. BoW + LogReg (tự code)            1.000   0.781      0.877        0.889
2. BoW + MLP (deep learning)         0.957   0.938      0.947        0.9473a. Encoder (MiniLM) + kNN           0.891   0.938      0.914        0.9763b. Encoder (MiniLM) + LogReg        0.957   0.917      0.936        0.9743c. Encoder (MiniLM) + MLP head      0.926   0.906      0.916        0.9474a. LLM nhỏ zero-shot                0.909   1.000      0.952        0.9524c. LLM nhỏ few-shot                 0.952   1.000      0.976        0.976

   train (s)  tin/s (cả lô)  p50 ms/tin  \
3. BoW + LogReg (tự code)            8.763      12423.965       0.046
4. BoW + MLP (deep learning)         6.419      41724.154       0.3243a. Encoder (MiniLM) + kNN           2.124         51.575      17.3993b. Encoder (MiniLM) + LogReg        3.438       1502.670       9.2703c. Encoder (MiniLM) + MLP head      3.593       1771.290       6.9844a. LLM nhỏ zero-shot                0.000          1.451     655.0704c. LLM nhỏ few-shot                 0.000          1.440     674.574

   p95 ms/tin  USD / 1k tin
5. BoW + LogReg (tự code)             0.101         0.000
6. BoW + MLP (deep learning)          0.397         0.000
   3a. Encoder (MiniLM) + kNN           21.646         0.000
   3b. Encoder (MiniLM) + LogReg         9.758         0.000
   3c. Encoder (MiniLM) + MLP head      12.304         0.000
   4a. LLM nhỏ zero-shot               955.802         0.029
   4c. LLM nhỏ few-shot                919.521         0.089
   • Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
   • SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
   • Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
   • train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
