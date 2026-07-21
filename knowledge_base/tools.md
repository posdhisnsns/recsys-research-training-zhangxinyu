# Markdown 语法

## 1、标题

**'#' + 空格，'#' 数量为几则为几级标题**

## 2、引用

> **使用'>' + 空格建立引用块**

## 3、列表

* **有序：数字 + '.' + 空格**

* **无序：'-' / '+' / '*' + 空格**

## 4、代码

**``使用反引号包裹建立代码``**

    使用四个空格建立代码块

## 5、分割线

使用多个'*' / '-' / '__' 建立分割线

***

## 6、链接

格式：`[超链接显示名](超链接地址 "超链接title")`

[Markdown 基本语法 | Markdown 教程](https://markdown.com.cn/basic-syntax/)

## 7、图片

格式：`![图片显示名](图片链接 "图片title")`。

![达达利亚](F:/ASUS/Pictures/达达利亚.jpg)

## 8、勾选框

格式：`- [ ] `

- [x] 已完成

# Git 用法

1. 设置全局用户名：`git config --global user.name "你的名字"`

2. 设置全局邮箱：`git config --global user.email "你的邮箱@example.com"`

3. 初始化：`git init`

4. 查看状态：`git status`

5. 将文件添加到暂存区：`git add .`

6. 提交更改到本地仓库：`git commit -m "xxx"`

7. 连接远程仓库：`git remote add origin https://github.com/你的用户名/仓库名.git`

8. 查看添加的远程仓库：`git remote -v`

9. 提交更改到远程仓库：`git push origin main`

10. 克隆远程仓库：`git clone https://github.com/用户名/仓库名.git`

11. 删除远程仓库：`git remote remove origin``

12. 从远程仓库拉取文件：`git fetch origin`

13. 从远程仓库拉取并合并文件：`git pull origin main`

14. 查看分支：`git branch`

15. 创建新分支：`git branch new-branch`

16. 切换分支：`git switch new-branch`

17. 合并分支：`git merge new-branch`

18. 删除分支：`git branch -d new-branch`

19. 查看本地与远程的差异：`git diff origin/main`

20. 查看提交历史：
    
        git log                    # 完整信息
        git log --oneline          # 简洁一行显示
        git log --oneline --graph  # 图形化显示分支结构

21. 查看文件改动对比：
    
        git diff                   # 工作区 vs 暂存区
        git diff --cached          # 暂存区 vs 上次提交
        git diff HEAD              # 工作区 vs 上次提交

22. 撤销：
    
        # 撤销 git add（文件从暂存区移回工作区）
        git reset HEAD file.txt
        
        # 撤销工作区的修改（恢复到上次提交的状态，会丢失未提交的改动）
        git restore file.txt
        
        # 修改上一次提交的说明（如果还没push）
        git commit --amend -m "新的提交信息"
        
        # 回退到某个历史版本（有三种模式）
        git reset --soft HEAD~1    # 撤销提交，改动保留在暂存区
        git reset --mixed HEAD~1   # 撤销提交，改动保留在工作区（默认）
        git reset --hard HEAD~1    # 彻底删除改动

# Conda 用法

## 环境管理

1. 创建新环境：`conda create --name myenv`

2. 创建指定 Python 版本的环境：`conda create --name myenv python=3.8`

3. 激活环境：`conda activate myenv`

4. 退出当前环境：`conda deactivate`

5. 查看所有环境：`conda env list`

6. 删除环境：`conda remove -n myenv --all`

## 包管理

1. 安装包：`conda install package_name`

2. 安装指定版本的包：`conda install package_name=1.2.3`

3. 更新包：`conda update package_name`

4. 卸载包：`conda remove package_name`

5. 查看已安装的包：`conda list`

## 其他常用命令

1. 查看帮助信息：`conda --help`

2. 查看 Conda 版本：`conda --version`

3. 搜索包：`conda search package_name`

4. 清理缓存和不需要的包：`conda clean --all`

## Jupyter Notebook 的使用（可选）

1. 安装 Jupyter Notebook：`conda install jupyter`

2. 启动 Jupyter Notebook：`jupyter notebook`

# 服务器使用

## 测试连接：`Test-NetConnection 222.198.156.91 -Port 22`

## 连接服务器：`ssh zhangxinyu@222.198.156.91`

## 📁 文件和目录操作命令

| 命令      | 作用            | 示例                                   |
| ------- | ------------- | ------------------------------------ |
| `pwd`   | 显示当前工作目录的绝对路径 | `pwd` → `/home/zhangxinyu`           |
| `ls`    | 列出目录内容        | `ls -lah`（显示详细信息、隐藏文件、人类可读大小）        |
| `cd`    | 切换目录          | `cd ~/research-training/lesson03`    |
| `mkdir` | 创建目录          | `mkdir -p`（递归创建父目录）                  |
| `touch` | 创建空文件或更新文件时间戳 | `touch notes.txt`                    |
| `cp`    | 复制文件或目录       | `cp notes.txt notes_copy.txt`        |
| `mv`    | 移动/重命名文件或目录   | `mv notes_copy.txt notes_backup.txt` |
| `rm`    | 删除文件或目录       | `rm -i`（删除前确认）                       |

---

## 📄 文件查看和编辑命令

| 命令     | 作用              | 示例                                    |
| ------ | --------------- | ------------------------------------- |
| `cat`  | 查看完整文件内容        | `cat notes.txt`                       |
| `head` | 查看文件开头部分（默认10行） | `head -n 2 notes.txt`（前2行）            |
| `tail` | 查看文件末尾部分（默认10行） | `tail -n 2 notes.txt`（后2行）            |
| `find` | 在目录树中查找文件       | `find . -maxdepth 1 -type f`（当前目录找文件） |

---

## 🔍 搜索和文本处理命令

| 命令       | 作用               | 示例                              |
| -------- | ---------------- | ------------------------------- |
| `grep`   | 在文件中搜索文本模式       | `grep -n "GPU" notes.txt`（显示行号） |
| `printf` | 格式化输出文本（可重定向到文件） | `printf "内容\n" > file.txt`      |

---

## 🖥️ 系统信息命令

| 命令         | 作用         | 示例                             |
| ---------- | ---------- | ------------------------------ |
| `whoami`   | 显示当前用户名    | `whoami` → `zhangxinyu`        |
| `hostname` | 显示主机名      | `hostname` → `lab-server`      |
| `uname`    | 显示系统信息     | `uname -m`（显示架构，如x86_64）       |
| `df`       | 显示磁盘空间使用情况 | `df -h ~`（人类可读格式显示home目录）      |
| `du`       | 显示文件/目录大小  | `du -sh .`（当前目录总大小）            |
| `history`  | 显示命令历史     | `history \| tail -n 20`（最后20条） |

---

## 🛠️ 进程和资源管理命令

| 命令           | 作用               | 示例                                                  |
| ------------ | ---------------- | --------------------------------------------------- |
| `ps`         | 显示进程状态           | `ps -u "$USER" -o pid,cmd`（显示用户进程）                  |
| `which`      | 显示命令的完整路径        | `which python` → `/home/user/miniconda3/bin/python` |
| `watch`      | 定期执行命令并全屏显示      | `watch -n 1 nvidia-smi`（每秒刷新GPU状态）                  |
| `nvidia-smi` | 显示NVIDIA GPU状态信息 | 查看GPU使用率、显存等                                        |

---

## 🌐 网络和下载命令

| 命令     | 作用           | 示例                                    |
| ------ | ------------ | ------------------------------------- |
| `wget` | 从网络下载文件      | `wget https://example.com/file.sh`    |
| `curl` | 传输数据（支持多种协议） | `curl -O https://example.com/file.sh` |
| `ssh`  | 远程登录服务器      | `ssh user@host`                       |

---

## 📦 包和环境管理命令

| 命令              | 作用        | 示例                                                                                                                                                  |
| --------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `conda`         | Conda包管理器 | `conda create -n env python=3.10`（创建环境）<br>`conda activate env`（激活环境）<br>`conda env list`（列出环境）<br>`conda clean -i`（清理索引缓存）<br>`conda config`（配置频道） |
| `python -m pip` | pip包管理器   | `pip install torch`（安装包）<br>`pip config set`（配置镜像源）                                                                                                 |

---

## 🚀 执行和权限管理命令

| 命令       | 作用            | 示例                              |
| -------- | ------------- | ------------------------------- |
| `chmod`  | 修改文件权限        | `chmod u+x hello.sh`（给用户添加执行权限） |
| `./`     | 执行当前目录下的可执行文件 | `./hello.sh`                    |
| `bash`   | 执行bash脚本      | `bash install.sh`               |
| `source` | 在当前shell中执行脚本 | `source ~/.bashrc`（重新加载配置文件）    |
| `export` | 设置环境变量        | `export LAB_AGENT_TOKEN`        |
| `unset`  | 删除环境变量        | `unset LAB_AGENT_TOKEN`         |

---

## 🔗 重定向和管道命令

| 符号   | 作用               | 示例                              |
| ---- | ---------------- | ------------------------------- |
| `>`  | 重定向输出到文件（覆盖）     | `echo "text" > file.txt`        |
| `>>` | 重定向输出到文件（追加）     | `echo "text" >> file.txt`       |
| `\|` | 管道，将前命令输出作为后命令输入 | `cat file \| grep "keyword"`    |
| `<<` | Here文档           | `cat > file <<'EOF'`（多行输入直到EOF） |

---

## 🧪 测试和比较命令

| 命令     | 作用         | 示例                                      |
| ------ | ---------- | --------------------------------------- |
| `cmp`  | 逐字节比较两个文件  | `cmp file1.csv file2.csv`（无输出表示相同）      |
| `test` | 测试文件属性或表达式 | `test -f ~/.condarc && cp ...`（文件存在则执行） |

---

## 📊 实用组合命令

| 命令组合                                           | 作用                           |
| ---------------------------------------------- | ---------------------------- |
| `ls -lah`                                      | 列出当前目录所有文件，显示权限、大小、修改时间      |
| `ps -u "$USER" -o pid,cmd --sort=pid \| head`  | 显示当前用户的进程ID和命令，按PID排序，只显示前几行 |
| `find . -maxdepth 2 -type f -print`            | 在当前目录下找文件，最多2层深度             |
| `history \| tail -n 20`                        | 显示最近20条命令历史                  |
| `RUN_DIR=$(ls -dt outputs/run_* \| head -n 1)` | 找到最新的运行目录并存入变量               |

---

## ⚠️ 特殊操作提示

1. **`~`** ：代表当前用户的home目录，如 `/home/zhangxinyu`

2. **`.`** ：代表当前目录

3. **`..`** ：代表父目录

4. **`-p`** 参数：`mkdir -p` 会递归创建所有不存在的父目录

5. **`-i`** 参数：`rm -i` 删除前会询问确认

6. **`-h`** 参数：`ls -lh`、`df -h` 等表示以人类可读格式显示大小（KB、MB、GB）
