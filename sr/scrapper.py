from google_images_search import GoogleImagesSearch

gis = GoogleImagesSearch('AIzaSyBeOSsw98I1aPkMbKme4QyiErm9LDWQOo8', 'a52bc9e8381dd4dd0')

_search_params = {
    'q': 'image of text',
    'num': 50,
    'fileType': 'jpg|png',
    'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived',
    'imgDominantColor': 'white', ##
    'imgColorType': 'gray' ##   
}

gis.search(search_params=_search_params, path_to_dir='./sr/dataset/')