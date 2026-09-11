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
> Chưa có API key thật nên chưa chạy bốn mức temperature để ghi nhận phản hồi. Về lý thuyết, temperature thấp thường cho câu trả lời ổn định hơn, còn temperature cao làm cách diễn đạt và nội dung đa dạng hơn nhưng có thể kém nhất quán. Cần bổ sung quan sát thực tế sau khi chạy; temperature thấp không bảo đảm thông tin đúng.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ bắt đầu với temperature = 0.2 để câu trả lời nhất quán và ít biến đổi khi giải đáp cùng một chính sách. Sau đó tôi sẽ thử trên các tình huống hỗ trợ thực tế để điều chỉnh; độ chính xác vẫn cần dựa vào tài liệu chuẩn và chuyển cho nhân viên khi thiếu thông tin.

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
> Khi chạy thật với Ollama `llama3.2:latest`, persona giáo viên dùng cách xưng hô “em” và ví dụ cuốn sổ được chia sẻ trong gia đình. Persona chuyên gia dùng các từ như giao dịch, node, cryptography và trình bày thành các bước đánh số. Cả hai phản hồi đều dài và bị ngắt giữa câu ở giới hạn `max_tokens=256`, nên lần thử này chưa cho thấy persona nào trả lời ngắn hơn khi hoàn chỉnh. System prompt làm thay đổi giọng văn, ví dụ và mức độ thuật ngữ; kết quả vẫn cần được kiểm chứng về nội dung.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn mẫu: “Mỗi buổi sáng, tôi đi bộ đến công viên gần nhà để tập thể dục. Không khí mát mẻ và tiếng chim hót giúp tôi cảm thấy thư giãn. Sau đó, tôi về nhà chuẩn bị bữa sáng cho gia đình rồi bắt đầu công việc. Hôm nay, tôi học cách xây dựng một ứng dụng trò chuyện bằng Python. Tôi thử gửi câu hỏi, đọc câu trả lời và đo thời gian phản hồi. Tôi cũng đếm token để hiểu cách ước tính chi phí, đồng thời so sánh kết quả giữa các cách diễn đạt khác nhau.”
>
> Đoạn này có 99 đơn vị theo `split()`. Chạy `count_tokens(text, model="gpt-4o")` với tiktoken thật được 121 token, còn ước lượng là 99 / 0.75 = 132 token; số đo thấp hơn ước lượng `(132 - 121) / 132 × 100 ≈ 8,33%`. Đây là tokenizer GPT-4o, không phải số token thực tế của Llama: cấu hình `llama3.2:latest` sẽ dùng nhánh dự phòng của hàm. Việc tách token phụ thuộc bộ mã hóa và đoạn văn; ký tự có dấu và các cụm tiếng Việt có thể bị tách thành nhiều token, nhưng không phải mỗi dấu thanh luôn là một token riêng. Ngoài ra, `split()` đếm các đơn vị cách nhau bởi khoảng trắng, không hoàn toàn tương đương số từ tiếng Việt; phép đo này chưa so sánh trực tiếp với bản tiếng Anh nên không chứng minh một tỷ lệ chung giữa hai ngôn ngữ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming hữu ích khi câu trả lời dài hoặc cần vài giây để sinh, chẳng hạn trợ lý viết nội dung, giải thích bài học hay tạo mã: người dùng thấy token đầu tiên sớm và có thể nhận ra hệ thống vẫn đang hoạt động. Nó cũng cho phép hiển thị tiến trình và dừng sớm khi câu trả lời đi sai hướng. Non-streaming phù hợp với phản hồi ngắn, thao tác cần kết quả hoàn chỉnh để xử lý tiếp như phân loại JSON, hoặc khi muốn kiểm tra toàn bộ nội dung trước khi hiển thị. Streaming không làm giảm tổng thời gian model chạy, chỉ giảm thời gian chờ cảm nhận.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giãn khoảng cách giữa các lần thử, giúp server có thời gian hồi phục và giảm tải dần khi nhiều request cùng thất bại. Với các khoảng 0,1; 0,2; 0,4 giây, những client bắt đầu retry gần nhau sẽ nhanh chóng lệch nhịp hơn so với delay cố định. Nếu hàng nghìn client đều chờ đúng một giây, chúng sẽ tạo một đợt request đồng loạt ngay khi server vừa hồi phục, gây quá tải lặp lại và có thể tạo vòng lặp “thundering herd”. Hệ thống thực tế thường thêm jitter ngẫu nhiên để phân tán thêm thời điểm retry.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt: “Bạn là trợ giảng thân thiện của khóa AI. Trả lời ngắn gọn, rõ ràng bằng tiếng Việt; giải thích thuật ngữ bằng ví dụ đơn giản; nếu không đủ thông tin thì nói rõ điều đó thay vì đoán.” Cụm “bằng tiếng Việt” giữ câu trả lời phù hợp với người học, còn “ngắn gọn, rõ ràng” hạn chế phần lan man trong một phiên chat giới hạn token. Yêu cầu nói rõ khi thiếu thông tin giúp người dùng phân biệt điều đã biết với suy đoán.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ giữ ba lượt gần nhất, nên trợ lý có thể quên yêu cầu hoặc quyết định được nêu ở đầu phiên. Tôi sẽ thêm bộ nhớ tóm tắt: trước khi cắt history, gửi các lượt cũ cho một hàm tóm tắt hoặc tự lưu các mục tiêu, ràng buộc và thông tin quan trọng vào một bản ghi ngắn. Mỗi lượt sau đó đưa bản tóm tắt cùng sáu message gần nhất vào system/context, đồng thời cập nhật tóm tắt khi cuộc hội thoại vượt giới hạn. Cách này giữ chi phí ổn định hơn việc gửi toàn bộ lịch sử nhưng cần kiểm tra để tránh tóm tắt sai.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
