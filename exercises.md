# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Tôi thấy càng nâng mức temperature thì câu trả lời có sự trau chuốt về ngôn ngữ và phong phú hơn
### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Với chatbot hỗ trợ khách hàng, tôi sẽ lựa chọn mức độ 0.4 đến 0.6 vì đây là mức độ trung bình khá tốt, đủ để truy xuất từ ngữ và đưa ra văn phong phù hợp cho khách hàng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Theo bảng giá có sẵn trong lab, mỗi ngày có 10.000 × 3 × 350 = 10.500.000 token đầu ra. Chi phí output của GPT-4o là 10.500.000 / 1.000 × 0.010 = 105 USD/ngày; GPT-4o-mini là 10.500.000 / 1.000 × 0.0006 = 6,30 USD/ngày, nên GPT-4o đắt khoảng 16,67 lần (chưa tính token đầu vào). GPT-4o có thể xứng đáng khi phân tích yêu cầu phức tạp từ nhiều tài liệu nếu đánh giá thực tế cho thấy chất lượng tốt hơn; mini phù hợp với phân loại yêu cầu hoặc trả lời FAQ đơn giản với số lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi 1 rất ngắn và dở dang, chỉ còn cụm tiếng Anh “chain, but misses the sharing/tamper-”, nên chưa thể hiện đầy đủ ví dụ đơn giản cho trẻ em. Phản hồi 2 cũng bị cắt, nhưng cụm “Chế độ đồng thuận” cho thấy hướng trả lời kỹ thuật hơn, phù hợp với persona chuyên gia tài chính. Trong lần chạy này, system prompt có ảnh hưởng đến hướng từ vựng nhưng giới hạn hoặc hành vi model khiến cả hai phản hồi chưa hoàn chỉnh. Khi phản hồi đầy đủ, persona giáo viên nên dùng ví dụ quen thuộc, còn persona chuyên gia nên tập trung vào blockchain phân tán, block, hash và cơ chế đồng thuận.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn mẫu: “Mỗi buổi sáng, tôi đi bộ đến công viên gần nhà để tập thể dục. Không khí mát mẻ và tiếng chim hót giúp tôi cảm thấy thư giãn. Sau đó, tôi về nhà chuẩn bị bữa sáng cho gia đình rồi bắt đầu công việc. Hôm nay, tôi học cách xây dựng một ứng dụng trò chuyện bằng Python. Tôi thử gửi câu hỏi, đọc câu trả lời và đo thời gian phản hồi. Tôi cũng đếm token để hiểu cách ước tính chi phí, đồng thời so sánh kết quả giữa các cách diễn đạt khác nhau.”
>
> Đoạn này có 99 từ. count_tokens bằng tiktoken đếm được 121 token, trong khi cách ước lượng 99 / 0.75 cho 132 token. Hai kết quả chênh nhau khoảng 8,33%, trong đó công thức ước lượng cao hơn. Tiếng Việt thường có thể tốn nhiều token hơn tiếng Anh vì từ có dấu và các cụm ký tự tiếng Việt đôi khi bị tokenizer tách thành nhiều phần; tuy nhiên mức chênh lệch phụ thuộc vào từng đoạn văn và tokenizer cụ thể.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming hữu ích khi câu trả lời dài hoặc cần vài giây để tạo ra. Nó cũng cho phép hiển thị tiến trình và dừng sớm khi câu trả lời đi sai hướng. Non-streaming phù hợp với phản hồi ngắn, thao tác cần kết quả hoàn chỉnh để xử lý tiếp như phân loại JSON, hoặc khi muốn kiểm tra toàn bộ nội dung trước khi hiển thị. Streaming không làm giảm tổng thời gian model chạy, chỉ giảm thời gian chờ cảm nhận.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần thời gian chờ giữa các lần thử, giúp API có thời gian phục hồi và giảm áp lực khi đang quá tải. Nếu hàng nghìn client đều retry sau đúng một giây, chúng sẽ gửi request đồng loạt, tạo hiệu ứng “thundering herd” và có thể khiến API tiếp tục quá tải. Vì vậy, hệ thống thực tế thường kết hợp exponential backoff với một khoảng jitter ngẫu nhiên để phân tán thời điểm retry.
---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn chế độ trợ giảng thân thiện. 
System prompt: “Bạn là trợ giảng thân thiện của khóa AI. Trả lời ngắn gọn, rõ ràng bằng tiếng Việt; giải thích thuật ngữ bằng ví dụ đơn giản; nếu không đủ thông tin thì nói rõ điều đó thay vì đoán.” Cụm “bằng tiếng Việt” giữ câu trả lời phù hợp với người học, còn “ngắn gọn, rõ ràng” hạn chế phần lan man trong một phiên chat giới hạn token. Yêu cầu nói rõ khi thiếu thông tin giúp người dùng phân biệt điều đã biết với suy đoán.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là lịch sử chỉ giữ ba lượt gần nhất, nên trợ lý có thể quên yêu cầu hoặc quyết định được nêu ở đầu phiên. Tôi sẽ cải thiện bằng cách tạo một bản tóm tắt ngắn chứa mục tiêu, yêu cầu và các quyết định chính trước. Ở mỗi lượt, trợ lý gửi bản tóm tắt cùng sáu tin nhắn gần nhất vào API và cập nhật bản tóm tắt khi cuộc trò chuyện dài hơn. Cách này giữ được ngữ cảnh quan trọng mà không làm chi phí input tăng theo toàn bộ lịch sử.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
