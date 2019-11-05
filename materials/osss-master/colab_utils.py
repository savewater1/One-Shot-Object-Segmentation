import os
from tqdm.auto import tqdm
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

class GoogleDrive:
  """
  Useful utils for accessing google drive from colab.
  This is useful when the gsuit drive account doesn't allow file streams, which is needed to mount the drive.
  """
  def __init__(self, creds=None):
    self.cred = creds or auth.authenticate_user()
    self.drive = build('drive', 'v3', credentials=self.cred)
  
  def _get_file_id(self, parent_id, name, is_folder=False):
    extra_params = ""
    if is_folder:
      extra_params = "and mimeType = 'application/vnd.google-apps.folder'"
    response = self.drive.files().list(q="'{}' in parents and name = '{}' {}".format(parent_id, name, extra_params),
                          spaces='drive',
                          fields='files(id)').execute()
    matches = response.get('files', [])
    if len(matches) != 1:
      if not matches:
        return None
      raise Exception("the provided path is ambiguous (duplicate folders with same names)!")
    return matches[0].get('id', None)
    
  def get_file_id(self, path, is_folder=False):
    paths = path.split('/')
    prev = 'root'
    for p in paths[:-1]:
      prev = self._get_file_id(prev, p, is_folder=True)
      if not prev:
        return None
    return self._get_file_id(prev, paths[-1], is_folder=is_folder)
  
  def download_file_with_id(self, file_id, dest_path):
    request = self.drive.files().get_media(fileId=file_id)
    with open(dest_path, 'wb') as fh:
      downloader = MediaIoBaseDownload(fh, request)
      done = False
      with tqdm(total=100) as pbar:
        prev_prog = 0
        while done is False:
            status, done = downloader.next_chunk()
            prog = int(status.progress() * 100)
            pbar.update(prog - prev_prog)
            prev_prog = prog
  
  def download_file(self, file_path, dest_path):
    file_id = self.get_file_id(file_path)
    if not file_id:
      raise Exception('unable to find file...')
    self.download_file_with_id(file_id, dest_path)
  
  def upload_file(self, src_path, location='root'):
    file_metadata = {
      'name': src_path.rsplit(os.sep, 1)[-1],
    }
    if location != 'root':
      parent = self.get_file_id(location)
      if not parent:
        raise Exception('Destination location not found!')
      file_metadata['parents'] = [parent]
    media = MediaFileUpload(src_path, resumable=True)
    request = self.drive.files().create(body=file_metadata,
                                        media_body=media,
                                        fields='id')
    response = None
    with tqdm(total=100) as pbar:
      prev_prog = 0
      while response is None:
        status, response = request.next_chunk()
        if status:
          prog = int(status.progress() * 100)
          pbar.update(prog - prev_prog)
          prev_prog = prog
      pbar.update(100-prev_prog)
    return response.get('id')
