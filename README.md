# nlp-course-mag-2026

## Submission Guidelines
1. Clone this repository
2. Create a folder with your full name (e.g., "john-doe")
3. Create subfolders for each assignment
4. Push your completed work before the deadline (for each assignment deadline is up to next lecture)
5. Never modify other students' folders

# How to Submit Your Homework

Follow these steps to submit your homework assignment via pull request.

## Prerequisites

- Install [Git](https://git-scm.com/downloads) on your computer
- Create a [GitHub account](https://github.com/join) if you don't have one
- Install a code editor (e.g., VS Code, Sublime Text)

## Step 1: Fork the Repository

1. Go to the course repository on GitHub
2. Click the **Fork** button in the top-right corner
3. This creates a copy of the repository under your GitHub account

## Step 2: Clone Your Fork

Open your terminal/command prompt and run:

```bash
git clone https://github.com/YOUR-USERNAME/REPOSITORY-NAME.git
cd REPOSITORY-NAME
```

Replace `YOUR-USERNAME` with your GitHub username and `REPOSITORY-NAME` with the actual repository name.

## Step 3: Create a New Branch

Create a branch for your homework. Use a descriptive name:

```bash
git checkout -b homework-1-yourname
```

Example: `git checkout -b homework-1-john-smith`

## Step 4: Complete Your Homework

1. Navigate to the appropriate homework folder
2. Complete your assignment
3. Save all your files

## Step 5: Commit Your Changes

Add your files to Git and commit them:

```bash
git add .
git commit -m "Complete Homework 1: [Brief description]"
```

Example: `git commit -m "Complete Homework 1: Implemented sorting algorithm"`

## Step 6: Push to GitHub

Push your branch to your fork on GitHub:

```bash
git push origin homework-1-yourname
```

## Step 7: Create a Pull Request

1. Go to your fork on GitHub
2. You'll see a yellow banner with "Compare & pull request" - click it
3. **Alternatively**: Go to the "Pull requests" tab and click "New pull request"
4. Make sure the base repository is the course repo and the head repository is your fork
5. Fill in the pull request template:
   - **Title**: `Homework 1 - Your Full Name`
   - **Description**: Mention what you implemented and any notes for the instructor
6. Click **Create pull request**

## Important Notes

- ⚠️ **Do not merge your own pull request** - your instructor will review and merge it
- Create a **separate branch** for each homework assignment
- Make sure your code runs without errors before submitting
- Include any necessary documentation or comments in your code
- Submit by the deadline specified in the assignment

## Need Help?

If you run into issues:
- Check the [GitHub documentation](https://docs.github.com)
- Ask questions in the course discussion forum
- Reach out during office hours

## Common Issues

**Problem: "Permission denied"**  
Solution: Make sure you forked the repository and are pushing to your own fork, not the main course repository.

**Problem: "Branch already exists"**  
Solution: Use a different branch name or delete the old branch: `git branch -d branch-name`

**Problem: "Merge conflicts"**  
Solution: Make sure you're working on a clean branch. Pull the latest changes from the main repository before starting.