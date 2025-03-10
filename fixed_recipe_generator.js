import fs from 'fs';
import fetch from 'node-fetch';
import path from 'path';

// Configuration
const OPENROUTER_API_KEY = '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';
const MODEL = 'deepseek/deepseek-r1-zero:free';
const PRODUCTS_CSV = 'heinz_products.csv';
const OUTPUT_DIR = 'knockoff_kitchen';
const HTML_DIR = `${OUTPUT_DIR}/html`;
const ASSETS_DIR = `${OUTPUT_DIR}/assets`;
const MAX_PRODUCTS = 3; // Limit for testing, increase as needed

// Main function
async function main() {
  try {
    // Create output directories
    createDirectories();
    
    // Read products from CSV
    const products = await readProductsFromCSV(PRODUCTS_CSV, MAX_PRODUCTS);
    console.log(`Found ${products.length} products to process`);
    
    // Generate recipe pages
    for (const product of products) {
      await generateRecipe(product.brand, product.product);
    }
    
    // Create index page
    createIndexPage(products);
    
    console.log('Recipe generation complete!');
    console.log(`Open ${HTML_DIR}/index.html to view the recipes`);
  } catch (error) {
    console.error(`Error in main function: ${error.message}`);
  }
}

// Create necessary directories
function createDirectories() {
  const dirs = [OUTPUT_DIR, HTML_DIR, ASSETS_DIR];
  
  for (const dir of dirs) {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`Created directory: ${dir}`);
    }
  }
}

// Read products from CSV
async function readProductsFromCSV(filePath, maxProducts) {
  try {
    const data = fs.readFileSync(filePath, 'utf8');
    const lines = data.split('\n');
    
    // Skip header and get product names
    const products = [];
    for (let i = 1; i < lines.length && products.length < maxProducts; i++) {
      const line = lines[i].trim();
      if (line) {
        // Extract brand name and product name
        let brandName = 'Heinz';
        let productName = line;
        
        // If product name contains the brand name, extract it
        if (line.includes('Heinz')) {
          productName = line.replace('Heinz', '').trim();
          // Remove leading spaces, commas, or hyphens
          productName = productName.replace(/^[\s,-]+/, '').trim();
        }
        
        products.push({
          brand: brandName,
          product: productName || line, // Fallback to full line if extraction fails
        });
      }
    }
    
    return products;
  } catch (error) {
    console.error(`Error reading products CSV: ${error.message}`);
    return [];
  }
}

// Create slug from text
function createSlug(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

// Generate a recipe
async function generateRecipe(brand, product) {
  const slug = createSlug(`${brand}-${product}`);
  const filePath = path.join(HTML_DIR, `${slug}.html`);
  
  // Skip if file already exists
  if (fs.existsSync(filePath)) {
    console.log(`Recipe page for ${brand} ${product} already exists, skipping`);
    return;
  }
  
  console.log(`Generating recipe for ${brand} ${product}`);
  
  // Generate recipe content using DeepSeek API
  const recipeContent = await generateRecipeContent(brand, product);
  
  // Create HTML file with modern design
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Homemade ${brand} ${product} | KnockoffKitchen.com</title>
  <style>
  /* KnockoffKitchen.com CSS */
  :root {
    --primary-color: #e52e2e;
    --primary-gradient: linear-gradient(135deg, #e52e2e, #b71c1c);
    --secondary-color: #333333;
    --light-gray: #f5f5f5;
    --medium-gray: #e0e0e0;
    --dark-gray: #666666;
    --white: #ffffff;
    --font-main: 'Roboto', Arial, sans-serif;
    --font-heading: 'Montserrat', Arial, sans-serif;
    --shadow: 0 2px 5px rgba(0,0,0,0.1);
    --card-shadow: 0 5px 15px rgba(0,0,0,0.1);
    --transition: all 0.3s ease;
  }
  
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  
  body {
    font-family: var(--font-main);
    line-height: 1.6;
    color: var(--secondary-color);
    background-color: var(--white);
  }
  
  a {
    color: var(--primary-color);
    text-decoration: none;
    transition: var(--transition);
  }
  
  a:hover {
    color: #b71c1c;
  }
  
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
  }
  
  /* Header */
  .site-header {
    background-color: var(--white);
    box-shadow: var(--shadow);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  
  .top-bar {
    background-color: var(--primary-color);
    color: var(--white);
    text-align: center;
    padding: 10px 0;
    font-weight: bold;
    letter-spacing: 1px;
  }
  
  .main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 0;
  }
  
  .logo {
    display: flex;
    align-items: center;
  }
  
  .logo-text {
    font-family: var(--font-heading);
    font-size: 24px;
    font-weight: 700;
    color: var(--secondary-color);
  }
  
  /* Navigation */
  .main-nav {
    background-color: var(--light-gray);
    padding: 10px 0;
    border-top: 1px solid var(--medium-gray);
    border-bottom: 1px solid var(--medium-gray);
  }
  
  .nav-list {
    display: flex;
    list-style: none;
    justify-content: center;
    flex-wrap: wrap;
  }
  
  .nav-item {
    margin: 0 15px;
  }
  
  .nav-link {
    color: var(--secondary-color);
    font-weight: 500;
    padding: 5px 0;
    position: relative;
  }
  
  .nav-link:hover {
    color: var(--primary-color);
  }
  
  /* Page Title */
  .page-title {
    background-color: var(--primary-color);
    color: var(--white);
    text-align: center;
    padding: 30px 0;
    margin-bottom: 40px;
  }
  
  .page-title h1 {
    font-family: var(--font-heading);
    font-size: 36px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
  }
  
  /* Recipe Cards */
  .recipes-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 30px;
    margin-bottom: 50px;
  }
  
  .recipe-card {
    background-color: var(--white);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: var(--card-shadow);
    transition: var(--transition);
    border: 1px solid var(--medium-gray);
  }
  
  .recipe-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 25px rgba(0,0,0,0.15);
  }
  
  .recipe-image {
    width: 100%;
    height: 200px;
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--dark-gray);
    font-style: italic;
    position: relative;
    overflow: hidden;
  }
  
  .recipe-image::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: var(--primary-gradient);
  }
  
  .recipe-content {
    padding: 25px;
  }
  
  .recipe-title {
    font-family: var(--font-heading);
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--secondary-color);
    position: relative;
    padding-bottom: 10px;
  }
  
  .recipe-title::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 50px;
    height: 3px;
    background: var(--primary-gradient);
    border-radius: 3px;
  }
  
  .recipe-meta {
    display: flex;
    justify-content: space-between;
    margin-bottom: 15px;
    font-size: 14px;
    color: var(--dark-gray);
    background-color: var(--light-gray);
    padding: 8px 12px;
    border-radius: 5px;
  }
  
  .recipe-brand i {
    margin-right: 5px;
    color: var(--primary-color);
  }
  
  .recipe-description {
    font-size: 15px;
    color: var(--dark-gray);
    margin-bottom: 20px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    line-height: 1.5;
  }
  
  .recipe-button {
    display: inline-block;
    background: var(--primary-gradient);
    color: var(--white);
    padding: 10px 20px;
    border-radius: 50px;
    font-weight: 500;
    transition: var(--transition);
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    text-align: center;
    width: 100%;
  }
  
  .recipe-button:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    transform: translateY(-2px);
    color: var(--white);
  }
  
  /* Recipe Detail */
  .recipe-detail {
    margin-top: 40px;
  }
  
  .recipe-header {
    margin-bottom: 30px;
  }
  
  .recipe-header h1 {
    font-family: var(--font-heading);
    font-size: 32px;
    font-weight: 700;
    color: var(--secondary-color);
    margin-bottom: 15px;
  }
  
  .recipe-image {
    width: 100%;
    height: 400px;
    background-color: var(--light-gray);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--dark-gray);
    font-style: italic;
    margin-bottom: 30px;
    border-radius: 8px;
  }
  
  .recipe-section {
    margin-bottom: 40px;
    background-color: var(--white);
    border-radius: 10px;
    box-shadow: var(--card-shadow);
    padding: 25px;
    transition: var(--transition);
  }
  
  .recipe-section:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
  }
  
  .recipe-section h2 {
    font-family: var(--font-heading);
    font-size: 24px;
    font-weight: 600;
    color: var(--secondary-color);
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--light-gray);
    display: flex;
    align-items: center;
  }
  
  .recipe-section h2::before {
    font-family: 'Font Awesome 5 Free';
    font-weight: 900;
    margin-right: 10px;
    color: var(--primary-color);
  }
  
  .recipe-section h2:contains('Introduction')::before { content: '\\f05a'; }
  .recipe-section h2:contains('Ingredients')::before { content: '\\f0c9'; }
  .recipe-section h2:contains('Instructions')::before { content: '\\f15c'; }
  .recipe-section h2:contains('Storage')::before { content: '\\f187'; }
  .recipe-section h2:contains('Variations')::before { content: '\\f074'; }
  .recipe-section h2:contains('Pro Tips')::before { content: '\\f0eb'; }
  .recipe-section h2:contains('Equipment')::before { content: '\\f0ad'; }
  .recipe-section h2:contains('Nutritional')::before { content: '\\f14a'; }
  .recipe-section h2:contains('Questions')::before { content: '\\f059'; }
  .recipe-section h2:contains('Serving')::before { content: '\\f2e7'; }
  .recipe-section h2:contains('Cost')::before { content: '\\f155'; }
  .recipe-section h2:contains('Conclusion')::before { content: '\\f15b'; }
  
  .recipe-meta-box {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border-radius: 10px;
    padding: 25px;
    margin-bottom: 40px;
    box-shadow: var(--card-shadow);
    border-left: 5px solid var(--primary-color);
  }
  
  .meta-item {
    display: flex;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid var(--medium-gray);
  }
  
  .meta-item:last-child {
    border-bottom: none;
  }
  
  .meta-label {
    font-weight: 600;
    display: flex;
    align-items: center;
  }
  
  .meta-label::before {
    font-family: 'Font Awesome 5 Free';
    font-weight: 900;
    margin-right: 10px;
    color: var(--primary-color);
  }
  
  .meta-item:nth-child(1) .meta-label::before { content: '\\f017'; } /* Prep Time */
  .meta-item:nth-child(2) .meta-label::before { content: '\\f2f1'; } /* Cook Time */
  .meta-item:nth-child(3) .meta-label::before { content: '\\f253'; } /* Total Time */
  .meta-item:nth-child(4) .meta-label::before { content: '\\f0f5'; } /* Yield */
  
  .ingredients-list, .instructions-list {
    list-style-position: outside;
    margin-bottom: 20px;
    padding-left: 20px;
  }
  
  .ingredients-list li {
    margin-bottom: 12px;
    padding: 8px 15px;
    background-color: var(--light-gray);
    border-radius: 5px;
    transition: var(--transition);
  }
  
  .ingredients-list li:hover {
    background-color: #e0e0e0;
    transform: translateX(5px);
  }
  
  .instructions-list li {
    counter-increment: step-counter;
    position: relative;
    padding: 15px 15px 15px 50px;
    margin-bottom: 25px;
    background-color: var(--light-gray);
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    transition: var(--transition);
  }
  
  .instructions-list li:hover {
    background-color: #e0e0e0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  }
  
  .instructions-list li::before {
    content: counter(step-counter);
    position: absolute;
    left: 10px;
    top: 50%;
    transform: translateY(-50%);
    background: var(--primary-gradient);
    color: var(--white);
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
  }
  
  /* Special styling for feature lists */
  .feature-list {
    list-style-type: none;
    padding-left: 0;
  }
  
  .feature-list li {
    position: relative;
    padding: 10px 15px 10px 35px;
    margin-bottom: 10px;
    background-color: var(--light-gray);
    border-radius: 5px;
    transition: var(--transition);
  }
  
  .feature-list li:hover {
    background-color: #e0e0e0;
    transform: translateX(5px);
  }
  
  .feature-list li::before {
    content: '\\f00c';
    font-family: 'Font Awesome 5 Free';
    font-weight: 900;
    position: absolute;
    left: 10px;
    color: var(--primary-color);
  }
  
  /* Special styling for comparison tables */
  .comparison-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    box-shadow: var(--card-shadow);
    border-radius: 10px;
    overflow: hidden;
  }
  
  .comparison-table th {
    background: var(--primary-gradient);
    color: var(--white);
    padding: 12px 15px;
    text-align: left;
  }
  
  .comparison-table tr:nth-child(even) {
    background-color: var(--light-gray);
  }
  
  .comparison-table tr:hover {
    background-color: #e0e0e0;
  }
  
  .comparison-table td {
    padding: 12px 15px;
    border-bottom: 1px solid var(--medium-gray);
  }
  
  /* Special styling for FAQ questions */
  .faq-question {
    background-color: var(--light-gray);
    padding: 15px;
    border-radius: 5px;
    margin-top: 20px;
    margin-bottom: 10px;
    font-weight: 600;
    color: var(--secondary-color);
    border-left: 4px solid var(--primary-color);
    transition: var(--transition);
  }
  
  .faq-question:hover {
    background-color: #e0e0e0;
    transform: translateX(5px);
  }
  
  .faq-question i {
    color: var(--primary-color);
    margin-right: 10px;
  }
  
  /* Special styling for pro tips */
  .pro-tip {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    padding: 15px;
    border-radius: 5px;
    margin-top: 20px;
    margin-bottom: 10px;
    font-weight: 600;
    color: var(--secondary-color);
    border-left: 4px solid #ffc107;
    transition: var(--transition);
  }
  
  .pro-tip:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
  }
  
  .pro-tip i {
    color: #ffc107;
    margin-right: 10px;
  }
  
  /* Special styling for serving suggestions */
  .serving-suggestion {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    padding: 15px;
    border-radius: 5px;
    margin-top: 20px;
    margin-bottom: 10px;
    font-weight: 600;
    color: var(--secondary-color);
    border-left: 4px solid #28a745;
    transition: var(--transition);
  }
  
  .serving-suggestion:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
  }
  
  .serving-suggestion i {
    color: #28a745;
    margin-right: 10px;
  }
  
  /* Footer */
  .site-footer {
    background-color: var(--secondary-color);
    color: var(--white);
    padding: 50px 0 20px;
    margin-top: 50px;
    text-align: center;
  }
  </style>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
</head>
<body>
  <header class="site-header">
    <div class="top-bar">
      THE MOST TRUSTED COPYCAT RECIPES
    </div>
    <div class="container">
      <div class="main-header">
        <a href="index.html" class="logo">
          <span class="logo-text">KnockoffKitchen.com</span>
        </a>
      </div>
    </div>
    <nav class="main-nav">
      <div class="container">
        <ul class="nav-list">
          <li class="nav-item"><a href="index.html" class="nav-link">Home</a></li>
        </ul>
      </div>
    </nav>
  </header>

  <main class="container">
    <div class="recipe-detail">
      <div class="recipe-header">
        <h1>Homemade ${brand} ${product}</h1>
      </div>
      
      <div class="recipe-image">
        [Recipe image will be added here]
      </div>
      
      <div class="recipe-meta-box">
        <div class="meta-item">
          <span class="meta-label">Prep Time:</span>
          <span>15 minutes</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Cook Time:</span>
          <span>30 minutes</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Total Time:</span>
          <span>45 minutes</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Yield:</span>
          <span>4 servings</span>
        </div>
      </div>
      
      ${recipeContent.replace(/\\boxed\{/g, '').replace(/\\boxed/g, '')}
    </div>
  </main>

  <footer class="site-footer">
    <div class="container">
      <p>&copy; 2025 KnockoffKitchen.com. All rights reserved.</p>
    </div>
  </footer>
</body>
</html>`;

  fs.writeFileSync(filePath, html);
  console.log(`Recipe page saved to: ${filePath}`);
}

// Generate recipe content using DeepSeek API
async function generateRecipeContent(brand, product) {
  try {
    // Prepare the prompt
    const prompt = `Generate a homemade copycat recipe for ${product} by ${brand}. This should closely replicate the original product using common household ingredients while allowing for customization and improved nutritional value.

### **Recipe Title:**
- The title should be catchy and contain relevant SEO keywords.
- Example: "Homemade ${brand} ${product}: A Copycat Recipe Better Than Store-Bought."

### **Introduction (SEO-Optimized)**
- Introduce the product and its history.
- Explain why people love it and why they might want to make it at home.
- Mention benefits of a homemade version (healthier, customizable, cost-effective).
- Include a **personal anecdote** or a **story** to make it engaging.

### **Recipe Details**
Provide the following metadata in a structured format:
- **Prep Time:** X minutes
- **Cook Time:** X minutes
- **Total Time:** X minutes
- **Yield:** X servings

### **Ingredients**
Provide a **detailed ingredient list** in both **US and metric measurements**, maintaining accuracy and proportions.
- Separate into different categories if applicable (e.g., dry vs. wet ingredients).
- Ensure measurements match what would be needed to mimic the store-bought product.

### **Instructions**
Step-by-step cooking instructions that:
1. Explain all necessary steps clearly.
2. Include cooking techniques and timing.
3. Specify the **ideal texture, consistency, and flavor development**.
4. Provide troubleshooting tips (e.g., what to do if it's too thick or too runny).

### **Storage Instructions**
- Explain how long the product lasts in the fridge, freezer, or pantry.
- Include preservation techniques for long-term storage.

### **Recipe Variations & Customization**
- Provide **alternative versions**, such as:
  - A **low-sugar or healthy version**.
  - A **spicy version**.
  - A **smoky version**.
  - Any relevant tweaks for dietary restrictions (e.g., vegan, gluten-free).

### **Special Equipment**
- List tools required (e.g., blender, saucepan, whisk).
- Mention any optional tools that can improve results.

### **Pro Tips**
Provide **3 expert-level cooking tips** to perfect the recipe.

### **Nutritional Comparison**
Create a **comparison table** between the homemade and store-bought version. Include:
- Calories
- Fat
- Sugar
- Sodium
- Protein
- Fiber
- Vitamins & minerals (if relevant)

### **Common Questions & Troubleshooting**
Include an **FAQ section** with 5-7 questions, such as:
- "Can I use fresh ingredients instead of processed ones?"
- "What can I do if the recipe turns out too sweet/salty?"
- "How do I make it last longer?"
- "Can I scale the recipe for large batches?"

### **Serving Suggestions**
List creative ways to use the homemade product, such as:
- Traditional uses (e.g., "Perfect as a dip for fries and burgers").
- Unique applications (e.g., "Use it as a marinade for BBQ ribs").

### **Cost Comparison**
Compare **homemade vs. store-bought cost per serving**:
- List approximate costs for each ingredient.
- Calculate total cost for a full batch.
- Show how much users can save by making it at home.

### **Conclusion**
- Summarize why this recipe is worth making.
- Encourage readers to share their results or leave a review.

Format the response as HTML with appropriate heading tags and structure.`;

    // Call the OpenRouter API
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'HTTP-Referer': 'https://knockoffkitchen.com',
        'X-Title': 'KnockoffKitchen Recipe Generator'
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          { role: 'system', content: 'You are a professional chef and food blogger who specializes in creating copycat recipes of popular store-bought products.' },
          { role: 'user', content: prompt }
        ],
        temperature: 0.7,
        max_tokens: 4000
      })
    });

    const data = await response.json();
    
    if (!data.choices || !data.choices[0] || !data.choices[0].message || !data.choices[0].message.content) {
      console.error(`Failed to generate recipe: ${JSON.stringify(data)}`);
      return getPlaceholderContent(brand, product);
    }
    
    // Clean up and format the content
    let content = data.choices[0].message.content;
    
    // Remove markdown code blocks and other unwanted tags
    content = content.replace(/```html/g, '');
    content = content.replace(/```/g, '');
    content = content.replace(/\(SEO-Optimized\)/g, '');
    
    // Format the content with more visual elements
    // Replace h2 tags with div.recipe-section > h2
    content = content.replace(/<h2>(.*?)<\/h2>/g, '<div class="recipe-section">\n<h2>$1</h2>');
    
    // Add closing div tags for recipe sections
    content = content.replace(/<div class="recipe-section">\s*<h2>(.*?)<\/h2>([\s\S]*?)(?=<div class="recipe-section">|$)/g, 
      '<div class="recipe-section">\n<h2>$1</h2>$2</div>\n\n');
    
    // Format tables with special styling
    content = content.replace(/<table>/g, '<table class="comparison-table">');
    
    // Add special styling for lists
    content = content.replace(/<ul>/g, '<ul class="feature-list">');
    
    // Add special styling for FAQ sections
    content = content.replace(/<h3>(.*?Question.*?)<\/h3>/gi, '<h3 class="faq-question"><i class="fas fa-question-circle"></i> $1</h3>');
    
    // Add special styling for tips
    content = content.replace(/<h3>(.*?Tip.*?)<\/h3>/gi, '<h3 class="pro-tip"><i class="fas fa-lightbulb"></i> $1</h3>');
    
    // Add special styling for serving suggestions
    content = content.replace(/<h3>(.*?Serving.*?)<\/h3>/gi, '<h3 class="serving-suggestion"><i class="fas fa-utensils"></i> $1</h3>');
    
    return content;
  } catch (error) {
    console.error(`Error generating recipe content: ${error.message}`);
    return getPlaceholderContent(brand, product);
  }
}

// Get placeholder content if API fails
function getPlaceholderContent(brand, product) {
  return `
  <div class="recipe-section">
    <h2>Introduction</h2>
    <p>${brand} ${product} is a beloved product known for its unique flavor and versatility. Making it at home allows you to customize the ingredients to your taste and dietary needs.</p>
  </div>
  
  <div class="recipe-section">
    <h2>Ingredients</h2>
    <ul class="ingredients-list">
      <li>Ingredient 1</li>
      <li>Ingredient 2</li>
      <li>Ingredient 3</li>
      <li>Ingredient 4</li>
      <li>Ingredient 5</li>
    </ul>
  </div>
  
  <div class="recipe-section">
    <h2>Instructions</h2>
    <ol class="instructions-list">
      <li>Step 1 description</li>
      <li>Step 2 description</li>
      <li>Step 3 description</li>
      <li>Step 4 description</li>
    </ol>
  </div>
  
  <div class="recipe-section">
    <h2>Storage Instructions</h2>
    <p>Store in an airtight container in the refrigerator for up to 2 weeks.</p>
  </div>
  
  <div class="recipe-section">
    <h2>Recipe Variations</h2>
    <p>Try these variations to customize the recipe to your taste:</p>
    <ul>
      <li><strong>Low-Sugar Version:</strong> Reduce sugar by half and add a natural sweetener.</li>
      <li><strong>Spicy Version:</strong> Add cayenne pepper or hot sauce for extra heat.</li>
      <li><strong>Gluten-Free Version:</strong> Substitute any gluten-containing ingredients.</li>
    </ul>
  </div>
  
  <div class="recipe-section">
    <h2>Pro Tips</h2>
    <ul>
      <li>Use high-quality ingredients for the best flavor.</li>
      <li>Allow flavors to develop by letting it sit overnight.</li>
      <li>Adjust seasonings to your personal taste preferences.</li>
    </ul>
  </div>`;
}

// Create index page
function createIndexPage(products) {
  const filePath = path.join(HTML_DIR, 'index.html');
  
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KnockoffKitchen.com - Copycat Recipes of Your Favorite Products</title>
  <style>
  /* KnockoffKitchen.com CSS */
  :root {
    --primary-color: #e52e2e;
    --primary-gradient: linear-gradient(135deg, #e52e2e, #b71c1c);
    --secondary-color: #333333;
    --light-gray: #f5f5f5;
    --medium-gray: #e0e0e0;
    --dark-gray: #666666;
    --white: #ffffff;
    --font-main: 'Roboto', Arial, sans-serif;
    --font-heading: 'Montserrat', Arial, sans-serif;
    --shadow: 0 2px 5px rgba(0,0,0,0.1);
    --card-shadow: 0 5px 15px rgba(0,0,0,0.1);
    --transition: all 0.3s ease;
  }
  
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  
  body {
    font-family: var(--font-main);
    line-height: 1.6;
    color: var(--secondary-color);
    background-color: var(--white);
  }
  
  a {
    color: var(--primary-color);
    text-decoration: none;
    transition: var(--transition);
  }
  
  a:hover {
    color: #b71c1c;
  }
  
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
  }
  
  /* Header */
  .site-header {
    background-color: var(--white);
    box-shadow: var(--shadow);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  
  .top-bar {
    background-color: var(--primary-color);
    color: var(--white);
    text-align: center;
    padding: 10px 0;
    font-weight: bold;
    letter-spacing: 1px;
  }
  
  .main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 0;
  }
  
  .logo {
    display: flex;
    align-items: center;
  }
  
  .logo-text {
    font-family: var(--font-heading);
    font-size: 24px;
    font-weight: 700;
    color: var(--secondary-color);
  }
  
  /* Navigation */
  .main-nav {
    background-color: var(--light-gray);
    padding: 10px 0;
    border-top: 1px solid var(--medium-gray);
    border-bottom: 1px solid var(--medium-gray);
  }
  
  .nav-list {
    display: flex;
    list-style: none;
    justify-content: center;
    flex-wrap: wrap;
  }
  
  .nav-item {
    margin: 0 15px;
  }
  
  .nav-link {
    color: var(--secondary-color);
    font-weight: 500;
    padding: 5px 0;
    position: relative;
  }
  
  .nav-link:hover {
    color: var(--primary-color);
  }
  
  /* Page Title */
  .page-title {
    background-color: var(--primary-color);
    color: var(--white);
    text-align: center;
    padding: 30px 0;
    margin-bottom: 40px;
  }
  
  .page-title h1 {
    font-family: var(--font-heading);
    font-size: 36px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
  }
  
  /* Recipe Cards */
  .recipes-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 30px;
    margin-bottom: 50px;
  }
  
  .recipe-card {
    background-color: var(--white);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: var(--card-shadow);
    transition: var(--transition);
    border: 1px solid var(--medium-gray);
  }
  
  .recipe-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 25px rgba(0,0,0,0.15);
  }
  
  .recipe-image {
    width: 100%;
    height: 200px;
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--dark-gray);
    font-style: italic;
    position: relative;
    overflow: hidden;
  }
  
  .recipe-image::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: var(--primary-gradient);
  }
  
  .recipe-content {
    padding: 25px;
  }
  
  .recipe-title {
    font-family: var(--font-heading);
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--secondary-color);
    position: relative;
    padding-bottom: 10px;
  }
  
  .recipe-title::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 50px;
    height: 3px;
    background: var(--primary-gradient);
    border-radius: 3px;
  }
  
  .recipe-meta {
    display: flex;
    justify-content: space-between;
    margin-bottom: 15px;
    font-size: 14px;
    color: var(--dark-gray);
    background-color: var(--light-gray);
    padding: 8px 12px;
    border-radius: 5px;
  }
  
  .recipe-brand i {
    margin-right: 5px;
    color: var(--primary-color);
  }
  
  .recipe-description {
    font-size: 15px;
    color: var(--dark-gray);
    margin-bottom: 20px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    line-height: 1.5;
  }
  
  .recipe-button {
    display: inline-block;
    background: var(--primary-gradient);
    color: var(--white);
    padding: 10px 20px;
    border-radius: 50px;
    font-weight: 500;
    transition: var(--transition);
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    text-align: center;
    width: 100%;
  }
  
  .recipe-button:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    transform: translateY(-2px);
    color: var(--white);
  }
  
  /* Footer */
  .site-footer {
    background-color: var(--secondary-color);
    color: var(--white);
    padding: 50px 0 20px;
    margin-top: 50px;
    text-align: center;
  }
  </style>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
</head>
<body>
  <header class="site-header">
    <div class="top-bar">
      THE MOST TRUSTED COPYCAT RECIPES
    </div>
    <div class="container">
      <div class="main-header">
        <a href="index.html" class="logo">
          <span class="logo-text">KnockoffKitchen.com</span>
        </a>
      </div>
    </div>
    <nav class="main-nav">
      <div class="container">
        <ul class="nav-list">
          <li class="nav-item"><a href="index.html" class="nav-link">Home</a></li>
        </ul>
      </div>
    </nav>
  </header>

  <div class="page-title">
    <div class="container">
      <h1>Copycat Recipes</h1>
    </div>
  </div>

  <main class="container">
    <p>Welcome to KnockoffKitchen.com! You've just found copycat recipes for all of your favorite famous foods! We show you how to easily duplicate the taste of iconic dishes and treats at home for less money.</p>
    
    <div class="recipes-grid">
      ${products.map(product => {
        const slug = createSlug(`${product.brand}-${product.product}`);
        return `<div class="recipe-card">
          <div class="recipe-image">[Recipe image]</div>
          <div class="recipe-content">
            <h3 class="recipe-title">Homemade ${product.brand} ${product.product}</h3>
            <div class="recipe-meta">
              <div class="recipe-brand"><i class="fas fa-tag"></i> ${product.brand}</div>
            </div>
            <p class="recipe-description">Make your own ${product.brand} ${product.product} at home with this easy copycat recipe. Healthier, customizable, and more affordable than store-bought.</p>
            <a href="${slug}.html" class="recipe-button"><i class="fas fa-utensils"></i> View Recipe</a>
          </div>
        </div>`;
      }).join('')}
    </div>
  </main>

  <footer class="site-footer">
    <div class="container">
      <p>&copy; 2025 KnockoffKitchen.com. All rights reserved.</p>
    </div>
  </footer>
</body>
</html>`;

  fs.writeFileSync(filePath, html);
  console.log(`Index page saved to: ${filePath}`);
}

// Run the main function
main().catch(console.error);
