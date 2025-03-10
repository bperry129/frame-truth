# Pushing to GitHub

Follow these steps to push your KnockoffKitchen project to GitHub:

## 1. Create a new repository on GitHub

1. Go to [GitHub](https://github.com/) and sign in to your account
2. Click on the "+" icon in the top right corner and select "New repository"
3. Name your repository (e.g., "knockoff-kitchen")
4. Add a description (optional)
5. Choose whether to make it public or private
6. Do NOT initialize the repository with a README, .gitignore, or license
7. Click "Create repository"

## 2. Push your local repository to GitHub

After creating the repository, GitHub will show you commands to push an existing repository. Run these commands from your project directory:

```bash
# Navigate to your project directory
cd knockoff-kitchen-project

# Add the remote repository URL
git remote add origin https://github.com/YOUR_USERNAME/knockoff-kitchen.git

# Push your code to GitHub
git push -u origin master
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## 3. Verify your repository

1. Go to `https://github.com/YOUR_USERNAME/knockoff-kitchen`
2. You should see all your project files there

## 4. Cloning on another computer

To continue working on this project from another computer:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/knockoff-kitchen.git

# Navigate to the project directory
cd knockoff-kitchen

# Install dependencies
npm install

# Run the recipe generator
node fixed_recipe_generator.js
```

## 5. Making changes

When you make changes on any computer:

```bash
# Add your changes
git add .

# Commit your changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

Then you can pull these changes on your other computer:

```bash
git pull
