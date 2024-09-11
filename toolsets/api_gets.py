import requests
import toolsets.chem_utils as cu
def get_lcb_data(method, splash, version='LCB2023'):
    # Base URL
    base_url = "https://api.metabolomics.us/cis/compound/{}/{}/{}".format(method, splash, version)
    
    # Headers with the API key
    headers = {
        "x-api-key": "lcb-fzkong-PeLfQ8Iuaz4PHwoDtBsc46"
    }
    
    # Make the GET request
    response = requests.get(base_url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()  # Return the JSON response
    else:
        print ({"error": "Request failed with status code {}".format(response.status_code)})
        return np.NAN
def format_method_string(input_string):
    input_string = input_string.strip()
    # Replace " | " with "%20%7C%"
    replaced_string = input_string.replace(" ", "%20")
    replaced_string = replaced_string.replace("|", "%7C")
    # Replace spaces with "%"
    
    return replaced_string