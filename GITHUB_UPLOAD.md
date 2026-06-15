# Uploading This Project To GitHub

## Option A: GitHub Web Upload

Use this if Git is not installed.

1. Go to <https://github.com/new>.
2. Create a repository, for example `local-literature-review-agent`.
3. Keep it public if you want others to use it.
4. Do not add a README, license, or `.gitignore` on GitHub because this folder already has them.
5. Click **uploading an existing file**.
6. Upload the project files and folders from this directory.

Do **not** upload:

- `.env`
- `.venv/`
- `runs/`
- `__pycache__/`

These are already ignored for Git users, but the web uploader may still show them if selected manually.

## Option B: Git Command Line

Install Git first:

<https://git-scm.com/download/win>

Then run:

```powershell
cd D:\Literature_Search
git init
git add .
git commit -m "Initial local literature review agent"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/local-literature-review-agent.git
git push -u origin main
```

## Option C: GitHub CLI

Install GitHub CLI:

<https://cli.github.com/>

Then run:

```powershell
cd D:\Literature_Search
gh auth login
gh repo create local-literature-review-agent --public --source=. --remote=origin --push
```
