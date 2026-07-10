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
