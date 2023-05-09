# Getting started

Paste the following into your terminal to set up the environment

```
conda env create -f environment.yml
conda activate langchain
```

Go to `constants.py` and paste in your OpenAI API key.

Then run the following to start messing around with it

```
python main.py
```

I'm mostly interested in this as a way to learn how to learn. If you have any ideas about a better form of back-and-forth, I want to know about them.

There are some known issues: Sometimes the retrieval chain returns too long of a prompt, and we get an `openai.error.InvalidRequestError`. Embarrassing.