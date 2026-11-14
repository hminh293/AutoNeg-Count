# third_party

Hai repo gốc gắn dưới dạng **git submodule**, ghim đúng commit đã đọc khi viết tài liệu:

| Thư mục | Nguồn | Commit |
|---|---|---|
| `CountGDPlusPlus/` | https://github.com/niki-amini-naieni/CountGDPlusPlus | `2ecbc43` |
| `CountSE/` | https://github.com/pppppz22/CountSE | `59c1e87` |

**Không commit thay đổi vào submodule.** Mọi sửa đổi viết thành `patches/NN_ten.patch` (tạo bằng
`git -C third_party/CountGDPlusPlus diff > patches/NN_ten.patch`), áp lúc cài bằng `environment/setup_colab.sh`.
