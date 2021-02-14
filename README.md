# ocr-with-bert
Improving quality of OCR with typo recognition and correction using pretrained BERT model.

# Installing tesseract

`$ sudo apt install tesseract-ocr`

`$ sudo apt install libtesseract-dev`

You can find installing tutorial under the following url: https://medium.com/quantrium-tech/installing-tesseract-4-on-ubuntu-18-04-b6fcd0cbd78f

Then, change the tesseract_cmd path in your code. Under Ubuntu, it should be:

`pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # config line`

And it's done! Now you can try it on your own images!

# Visualizing results

In order to plot the example results of OCR model, run `draw.py` function. You can test on the single image or on the whole directory, e.g.

`python draw.py examples/1.png`

or

`python draw.py examples/`
