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

### Predict Expr 1B

Mình khá là expect với learning rate cực cao thì model sẽ bị overshoot và không thể hội tụ được, tuy nhiên model vẫn có thể đạt được loss rất nhỏ. Mình chỉ implement các hàm dự đoán và không rõ optimizer model đang dùng là gì để có dự đoán chính xác hơn

### Predict Expr 3B

Mình khá là mong sự chênh lệch giữa shallow và deep learning model sẽ lớn hơn, tuy nhiên với tập dataset nhỏ và không quá phức tạp thì mình phải chấp nhận rằng không có quá nhiều không gian để các mô hình deep learning có thể có cách biệt rõ hơn với shallow model

### Predict Expr 5

Mình khá là expect Embedding + kNN sẽ có Pareto tốt hơn so với Embedding + LogRes. Vì tập dataset không quá phức tạp nên mình nghĩ với lợi thế tìm kiếm rất nhanh của kNN sẽ có lợi hơn LogRes. Thực tế thì với tập dataset khá nhỏ, những kết quả có thể biến động nhiều hơn.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

Model tụt nhiều nhất là BoW + MLP, giảm 0.579 điểm F1. Tập test thường quá lạc quan vì thường có cùng phân phối với dữ liệu huấn luyện, chứa các từ khóa spam rõ ràng và ít trường hợp né từ khóa, lừa đảo tinh vi hoặc prompt injection. 50 case khó tạo ra sự thay đổi phân phối và kiểm tra khả năng tổng quát hóa thực tế.

Với BoW, nhóm khó nhất là spam giọng tự nhiên và thao túng/lừa đảo. BoW chỉ nhìn các token rời rạc và tần suất từ, nên dễ bỏ sót spam không chứa từ khóa rõ ràng hoặc dùng cách diễn đạt giống tin nhắn bình thường. Nhóm né từ khóa cũng khó vì chỉ cần thay đổi cách viết là vector từ thay đổi mạnh.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*



Với `message[0]` và `message[1]`, mình nhắm đến việc chỉ trigger Embedding, BoW sẽ không thể phát hiện `spam` vì mình đã cố gắng né những từ có thể có signal, nhất là với nhóm có signal mạnh (vd “free”,  “win”, “call”, “txt”). Tuy nhiên câu từ vẫn có đủ tín hiệu để embedding có thể xác định được ngữ cảnh và mục đích giao tiếp của câu

Với `message[2]`, mình nhắm đến việc BoW sẽ bị trigger vì từ "free", và Embedding sẽ cho kết quả `ham`. ở đây "free" có tín hiệu mạnh và những từ xung quanh câu chưa đủ nhiều để BoW có thẻ classify là `ham`

Lần này với `message[3]`, mình muốn cả 2 mô hình đều trả kết quả `ham`. Với embedding có thể nhận biết được ngữ nghĩa, và BoW có thêm nhiều từ xung quanh trong câu để tín hiệu của "free" không bị lấn át làm đổi nhãn dự đoán của mô hình

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

Ta có thể sử dụng một pipeline phân loại theo độ khó, chia theo từng tier xử lý và sẽ escalate (như cơ chế của SOC) lên tầng cao hơn mỗi khi gặp input khó
1/ Dùng các luật đơn giản để xử lý các case rõ ràng, chẳng hạn URL độc hại, người gửi bị chặn hoặc mẫu spam quá rõ.
2/ Đưa các tin còn lại vào mô hình Embedding + Logistic Regression.
3/ Nếu độ tin cậy cao, ví dụ \(p < 0,3\) hoặc \(p > 0,7\), dùng luôn dự đoán của mô hình nhanh.
4/ Nếu mô hình không chắc chắn, với \(p \in [0,3; 0,7]\), chuyển tin nhắn sang LLM để kiểm tra thêm.
5/ Lưu lại các case không chắc chắn để theo dõi và huấn luyện lại sau này.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

Mình sử dụng AI chỉ để fix bug và typo trong code của mình. Sau mỗi lần đọc docs, mình muốn tự implement code vì mình khác thích code tay, và mình nhận ra mình rất dễ mắc các lỗi typo không mong muốn, AI rất nhạy để phát hiện ra những lỗi này.
