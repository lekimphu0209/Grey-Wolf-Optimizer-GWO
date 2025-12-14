# Grey Wolf Optimizer (GWO) - MATLAB Demo  
📌 **Repo:** https://github.com/LuongHuongGiang20236027/group14_gwo

---

## 📘 Giới thiệu  
Đây là chương trình demo thuật toán **Grey Wolf Optimizer (GWO)** viết bằng **MATLAB**, phục vụ cho bài tập môn *Nhập môn Kỹ thuật Truyền thông*.  

Thuật toán GWO mô phỏng hành vi săn mồi của bầy sói xám với ba cấp:  
- **Alpha** – nghiệm tốt nhất  
- **Beta** – nghiệm tốt thứ hai  
- **Delta** – nghiệm tốt thứ ba  
- Các sói còn lại di chuyển theo ba con dẫn đầu để tiến tới nghiệm tối ưu.

---

## 🧠 Bài toán tối ưu minh họa  
Sử dụng hàm **Sphere** – một hàm chuẩn trong tối ưu hóa:

\[
f(x) = \sum_{i=1}^{n} x_i^2
\]

- Nghiệm tối ưu tại: \( x = 0 \)  
- Đơn giản, phù hợp để kiểm thử khả năng hội tụ của thuật toán meta-heuristic.

---

## 📂 Danh sách file  
| File | Mô tả |
|------|-------|
| **gwo_demo.m** | Mã nguồn MATLAB thuật toán GWO |
| **README.md** | Tài liệu mô tả và hướng dẫn sử dụng |

---

## ▶ Cách chạy chương trình
1. Mở **MATLAB**  
2. Chọn thư mục chứa file `gwo_demo.m`  
3. Chạy:

```matlab
gwo_demo



