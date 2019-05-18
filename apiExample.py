import cv2
import requests
import json
import argparse

api_url_base = 'http://127.0.0.1:5000/api'


img_path = './images/testing-images/pepsi+products000029.jpg'
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=img_path, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())

img_path = args['image']

img = cv2.imread(img_path)
if img is None:
    exit('Image path is not valid. Please check!')

def detect_logo(files):
    api_url = '{0}/detect_logo'.format(api_url_base)

    response = requests.post(api_url, data=files)

    return response


def main():
    print('Requesting server...')
    img_parsed = cv2.imencode(img_path[-4:], img)[1].tostring()
    # img_encoded = base64.b64encode(img_parsed)

    # files = {'image': img_encoded}
    query_response = detect_logo(img_parsed)
    res = json.loads(query_response.content.decode('utf-8'))

    with open('detectedBoxesAPI.csv', 'a') as f:
        print('\nDeteced Boxes Coordinates: ')
        num_boxes = 0
        f.write("\nFor image at: {0}\n".format(img_path))
        for entry in res['detectedBoxes']:
            num_boxes += 1
            # Write on file
            for i in entry:
                f.write(str(i)); f.write(" ")
            f.write('\n')
            print(entry)
        f.write("Number of Boxes: {0:d}\n".format(num_boxes))
        print('\nNumber of Boxes detected: {0}\n'.format(num_boxes))


if __name__ == '__main__':
    main()
