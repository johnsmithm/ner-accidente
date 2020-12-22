import gdown

url = 'https://drive.google.com/uc?id=1TKxABVonvgiE2oBgTKB6oOxXkynqjFth'
output = 'data/raw/raw_data.csv'
gdown.download(url, output, quiet=False)