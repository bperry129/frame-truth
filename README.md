# KnockoffKitchen Recipe Generator

A Node.js application that generates copycat recipes for popular products using the DeepSeek AI model via OpenRouter.

## Features

- Generates detailed copycat recipes for popular products
- Creates a modern, responsive website with proper styling
- Includes recipe variations, storage instructions, and pro tips
- Uses AI to generate SEO-optimized content

## Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/knockoff-kitchen.git
   cd knockoff-kitchen
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Set up your OpenRouter API key:
   - Get an API key from [OpenRouter](https://openrouter.ai/)
   - Update the `OPENROUTER_API_KEY` in `fixed_recipe_generator.js`

## Usage

1. Add product names to `heinz_products.csv` (one product per line)

2. Run the recipe generator:
   ```
   node fixed_recipe_generator.js
   ```

3. View the generated website:
   ```
   open knockoff_kitchen/html/index.html
   ```

## Project Structure

- `fixed_recipe_generator.js` - Main script that generates recipes and website
- `heinz_products.csv` - List of products to generate recipes for
- `knockoff_kitchen/` - Generated website (HTML, CSS, assets)

## Customization

- Modify the CSS in `fixed_recipe_generator.js` to change the website design
- Adjust the prompt in `generateRecipeContent()` to change the recipe format
- Update the `MAX_PRODUCTS` constant to generate more recipes

## License

MIT
