# 🚀 Push Eventformer to GitHub & Run on Colab

## Step 1: Install Git (if not installed)

Download Git from: https://git-scm.com/download/win

Or use winget:
```powershell
winget install Git.Git
```

After installation, **restart your terminal** or VS Code.

---

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `Eventformer`
3. Description: `Frame-Free Vision Transformer for Event Cameras`
4. Choose: **Public** (so Colab can access it)
5. **DO NOT** initialize with README (we have our own)
6. Click **Create repository**

---

## Step 3: Push Code to GitHub

Open a **new terminal** and run these commands:

```powershell
# Navigate to project folder
cd "c:\Users\jkina\Downloads\visrecCurrent\Eventformer"

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Eventformer - Frame-Free Vision Transformer"

# Add your GitHub repo as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/Eventformer.git

# Push to GitHub
git branch -M main
git push -u origin main
```

If prompted for credentials, use your GitHub username and a Personal Access Token:
- Go to GitHub → Settings → Developer settings → Personal access tokens → Generate new token
- Select scopes: `repo`
- Copy the token and use it as your password

---

## Step 4: Run on Google Colab

### Option A: Use the included Colab notebook

1. Go to your GitHub repo: `https://github.com/YOUR_USERNAME/Eventformer`
2. Click on `Eventformer_Colab.ipynb`
3. Click "Open in Colab" badge at the top

### Option B: Create new Colab notebook

1. Go to https://colab.research.google.com
2. File → New notebook
3. Runtime → Change runtime type → GPU (T4)
4. Run these cells:

```python
# Cell 1: Clone your repo
!git clone https://github.com/YOUR_USERNAME/Eventformer.git
%cd Eventformer/code

# Cell 2: Install dependencies
!pip install -q einops timm h5py tensorboard seaborn

# Cell 3: Test the model
!python main.py --mode test

# Cell 4: Quick training test (uses synthetic data)
!python train.py --dataset ncaltech101 --model tiny --epochs 5 --batch_size 16

# Cell 5: Run ablation study
!python ablation.py --dataset ncaltech101 --model_size tiny --epochs 10 --num_runs 2

# Cell 6: Generate figures
!python visualize.py --output_dir ../figures
```

---

## Step 5: Save Results

Mount Google Drive to save checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')

# Save outputs
!cp -r ../outputs /content/drive/MyDrive/Eventformer_outputs/
```

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Test all modules | `python main.py --mode test` |
| Train (tiny model) | `python train.py --dataset ncaltech101 --model tiny --epochs 100` |
| Train (small model) | `python train.py --dataset ncaltech101 --model small --epochs 100` |
| Ablation study | `python ablation.py --dataset ncaltech101 --epochs 20` |
| Evaluate | `python evaluate.py --checkpoint outputs/best_model.pth` |
| Generate figures | `python visualize.py --output_dir figures/` |

---

## Colab GPU Tips

- **Free Tier**: T4 GPU, ~12GB RAM, 12-hour session limit
- **Colab Pro**: A100 GPU, ~40GB RAM, longer sessions
- If you run out of memory, reduce `--batch_size` or `--num_events`
- Mount Google Drive to persist checkpoints between sessions

---

## File Structure on GitHub

```
Eventformer/
├── 📄 Eventformer_Colab.ipynb    ← Open this in Colab!
├── 📄 eventformer.tex            ← Paper source
├── 📄 eventformer.bib            ← Bibliography
├── 📄 .gitignore
├── 📁 code/
│   ├── main.py                   ← Entry point
│   ├── train.py
│   ├── evaluate.py
│   ├── ablation.py
│   ├── visualize.py
│   ├── requirements.txt
│   ├── README.md
│   ├── 📁 models/
│   ├── 📁 datasets/
│   └── 📁 configs/
```
