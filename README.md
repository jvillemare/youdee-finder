# YoUDee Finder
---

Locating the University of Delaware mascot, [YoUDee](https://en.wikipedia.org/wiki/YoUDee), is a matter of grave importance.

![YoUDee laying on Raymond Field](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/YoUDee_Laying_Down.jpg/1024px-YoUDee_Laying_Down.jpg)

Bird watching allows us to gain insights into the ecological health of our surroundings. The Delaware Blue Hen is a rare species only found at the University of Delaware.

To aid Delawarean ornithologists, this Machine Learning (ML) model was developed.

## Statistics

The ML model was trained using:
 - `1,165` samples of YoUDee images.
 - `2,470` samples of neutral, non YoUDee images.
  	- Neutral samples include other mascots, images of places where YoUDee would be but isn't (football, basketball games), and fursuiters (for their similarity to costumed mascots.)
 - `4.37 gigabytes` of image data.
 - `632` of the samples were scrapped from YoUDee and other mascot Instagram accounts using [rarcega's scripts](https://github.com/rarcega/instagram-scraper).

Some examples of the samples can be [seen here](samples.md). The full sample data set cannot be provided here due to privacy and copyright concerns.

## Installation

Ensure you are running at least Python 3.7 by typing:

```
python --version
```

Clone this GitHub repo:

```
git clone https://github.com/jvillemare/youdee-finder.git
```

Install the requirements:

`pip install requirements.txt` or `python -m pip install requirements.txt`


## Usage

Below are the few basic command line flags you can set for this script.

#### Find in a single image
```
python find_youdee.py --image=IMG_0001.jpg
```

#### Find in a single video
```
python find_youdee.py --video=IMG_0001.mov
```

#### Find in a directory
```
python find_youdee.py --dir=pictures/
```

Non-recursively searches a directory for all supported image and video files, then processes and makes all the predictions for them.

#### Write output to a CSV
```
python find_youdee.py --dir=pictures/ --output=predictions.csv
```

Works for both single images and videos, or a full directory.

CSV is in the format of:
 - `type`: `image` or `video`.
 - `filename`: Filename of image/video.
 - `frame_count`: Number of frames in file (images always have `1`).
 - `youdee_result`: Confidence that YoUDee is in image as a decimal percent (i.e. `0.50` = 50%).
 - `neutral_result`: Confidence that YoUDee is not in an image as a decimal percent.
 - `has_youdee`: `True` or `False` boolean, whether or not the confidence of YoUDee was high enough to be significant.

## Other Notes

For videos, YoUDee could be in part of a video, but not other parts. Since we're concerned with finding any instance of YoUDee, if the confidence of YoUDee being in a video is high enough for a consecutive number of frames, then YoUDee is considered to be in that video.
