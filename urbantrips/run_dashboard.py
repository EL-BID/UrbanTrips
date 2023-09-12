import os
import requests

def download_githubfile(url_file, file_name):

    if not os.path.isfile(file_name):
    
        # Send a request to get the content of the image file
        response = requests.get(url_file)
    
        # Save the content to a local file
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print('copiado: ', file_name)

def main():
    
    os.makedirs('urbantrips/dashboard/pages', exist_ok=True)
    download_githubfile(url_file = 'https://raw.githubusercontent.com/EL-BID/UrbanTrips/dev/urbantrips/dashboard/dashboard.py', 
                        file_name = 'urbantrips/dashboard/dashboard.py')
    download_githubfile(url_file = 'https://raw.githubusercontent.com/EL-BID/UrbanTrips/dev/urbantrips/dashboard/pages/1_Datos Generales.py', 
                        file_name = 'urbantrips/dashboard/pages/1_Datos Generales.py')
    download_githubfile(url_file = 'https://raw.githubusercontent.com/EL-BID/UrbanTrips/dev/urbantrips/dashboard/pages/2_Indicadores de oferta.py', 
                        file_name = 'urbantrips/dashboard/pages/2_Indicadores de oferta.py')
    
    print('')
    print('Debe correr desde la terminal streamlit run urbantrips/dashboard/dashboard.py')
    print('')
    # !streamlit run urbantrips/dashboard/dashboard.py

if __name__ == "__main__":
    main()