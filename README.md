# cmsc426-final-project

### Setup stuff
1. python3.10 -m venv venv (might need to specify version later for tensor flow)
2. source venv/bin/activate (mac)
3. pip3 install -r requirements.txt
### Hand Gesture Stuff:
1. need a labeled kaggle dataset of image hand gestures
2. preprocess data --> resize images, normalize pixel values, Encode the labels
3. Build and train the CNN

### Actual game part:
1. computer chooses choice randomly from rock paper scissors
2. says 3 2 1 and user needs to have chosen by the end of it
3. takes picture and figure out what user chose
4. if above confidence level (we choose) compare to computers and say who won 
5. if lower than confidence level its nothing
6. while all this happens keep track of what the user chooses and when they don't choose anything, list out the confidence levels of the options
7. idk

### Extra stuff: 
1. easy mode (random), hard mode (based on stats of what the user has inputted)
