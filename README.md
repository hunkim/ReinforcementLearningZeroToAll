# Lab code (WIP), but call for comments

This is work in progress, and may have bugs. (lab-08 is in WIP.)
However, we call for your comments and pull requests. Check out our style guide line:

* More TF (1.0) style: use more recent and decent TF APIs.
* More Pythonic: fully leverage the power of python
* Readability (over efficiency): Since it's for instruction purposes, we prefer *readability* over others.  
* Understandability (over everything): Understanding TF key concepts is the main goal of this code.
* KISS: Keep It Simple Stupid! https://www.techopedia.com/definition/20262/keep-it-simple-stupid-principle-kiss-principle
 
## File naming rule:

* lab-XX-X-[name].py: TensorFlow lab code

## How to use uploader:
* Go to https://gym.openai.com/
* Login with your github account
    * https://gym.openai.com/users/YOUR_GITHUB_ACCOUNT
* Grab the api key from  
![user](assets/openai_user.jpg)
* Modify `gym.ini`
* In console
```bash
python gym_uploader.py gym-results/
```

## Install requirements
```bash
pip install -r requirements.txt
```

## Run test and autopep8
TODO: Need to add more test cases

```bash
pytest

# http://stackoverflow.com/questions/14328406/
pip install autopep8 # if you haven't install
autopep8 . --recursive --in-place --pep8-passes 2000 --verbose --ignore E501
```
## Automatically create requirements.txt

```bash
pip install pipreqs

pipreqs /path/to/project
```
http://stackoverflow.com/questions/31684375

## Contributions/Comments
We always welcome your comments and pull requests.
