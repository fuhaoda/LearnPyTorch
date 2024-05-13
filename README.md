# Learn PyTorch

The purpose of this repo is to create education materials on PyTorch for model fitting.
The materials are oragnized as course number from 101, 201, etc..

Here are some summaries
- 101 We start from using a numpy to fit a linear model, then we use PyTorch utilities to simplify the work
- 201 This is the MINST example building a deep neural network for hand writting zip code recognition. 
- 301 Seq classification and seq2seq models
- 401 SDE diffusion model on Generative AI


# Setting Up a Python Virtual Environment for Deep Learning

This guide will walk you through setting up a Python virtual environment on your local machine. This setup allows you to manage dependencies and ensure that the deep learning code from this course runs smoothly in Jupyter Notebook.

## Prerequisites

Before you begin, make sure you have Python installed on your system. You can download Python from python.org. This course assumes you are using Python 3.

## Step 1: Create a Virtual Environment

First, open your terminal and navigate to your project directory or where you want to set up your virtual environment.

`cd path/to/your/project-directory`

Create a virtual environment named env_name by running:

`python3 -m venv env_name`

Replace env_name with your preferred name for the virtual environment.

## Step 2: Activate the Virtual Environment

Activate the virtual environment using the command below:

On Windows:

`.\env_name\Scripts\activate`

On MacOS and Linux:

`source env_name/bin/activate`

You should see the name of your virtual environment in parentheses on your terminal prompt, indicating that it is active.

## Step 3: Install Required Packages

With the virtual environment activated, install the necessary Python packages for the course:

`pip install torch jupyterlab jupyter`

This command installs PyTorch, JupyterLab, and Jupyter. You can modify this command to include any other packages you need for the course.

## Step 4: Start JupyterLab

After installing the packages, you can start JupyterLab by running:

`jupyter lab`

This command will start the JupyterLab server, and you should see a link in your terminal that you can open in a web browser to access JupyterLab.

## Conclusion

You now have a fully functional Python virtual environment with JupyterLab, ready to run the deep learning code provided in this course. If you encounter any issues, ensure that your virtual environment is activated and that you've installed all required packages.

