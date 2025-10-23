import os
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

EXPERIMENT_DIR = 'HDSAC Experiments'
SCOPES = ["https://www.googleapis.com/auth/drive"]
try:
    CLIENT_SECRET_FILE = os.environ["G_DRIVE_SECRET_PATH"]
except KeyError:
    print("Error: G_DRIVE_SECRET_JSON environment variable not set.")

class DriveService():

    def __init__(self):
        self._creds = self._get_credentials()
        self._service = build("drive", "v3", credentials=self._creds)
        self._experiment_fold_id = self.get_folder_id_by_name(EXPERIMENT_DIR)

    def create_folder(self, folder_name : str, parent_id : str=None) -> str:
        """Will create the folder in google drive

        Args:
            folder_name (str): The folder name
            parent_id (str, optional): The Parent ID. Will default to the EXPERIMENT DIR. Defaults to None.

        Returns:
            str: Folder ID
        """
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }

        if parent_id:
            folder_metadata['parents'] = [parent_id]
        else:
            folder_metadata['parents'] = [self._experiment_fold_id]

        folder = self._service.files().create(
            body=folder_metadata,
            fields='id, name'
        ).execute()

        print(f"Created folder: {folder.get('name')} (ID: {folder.get('id')})")
        return folder.get('id')

    def upload_file_to_folder(self, folder_id : str, file_path : str) -> str:
        """Will upload the local file to the google drive folder

        Args:
            folder_id (str): Folder ID to upload to
            file_path (str): Local file path of file to be uploaded

        Returns:
            str: File ID
        """
        file_name = os.path.basename(file_path)
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }

        media = MediaFileUpload(file_path, resumable=True)
        uploaded_file = self._service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, parents'
        ).execute()

        print(f"Uploaded '{uploaded_file.get('name')}' to folder ID: {folder_id}")
        return uploaded_file.get('id')

    def get_folder_id_by_name(self, folder_name : str) -> str:
        """Will get the Google Drive ID of a specific folder name if it can be found

        Args:
            folder_name (str): Folder Name

        Raises:
            ValueError: When folder cannot be found

        Returns:
            str: Folder ID
        """
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self._service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = results.get('files', [])
        
        if not folders:
            raise ValueError(f'Folder: {folder_name} not found')
        
        folder_id = folders[0]['id']
        print(f"Found folder '{folder_name}' with ID: {folder_id}")
        return folder_id

    def _get_credentials(self):
        creds = None
        if os.path.exists("./g_drive/token.pkl"):
            with open("./g_drive/token.pkl", "rb") as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
                creds = flow.run_local_server(port=8080)
            with open("token.pkl", "wb") as token:
                pickle.dump(creds, token)
        return creds

