from queue import Queue
from threading import Thread








def train_model():
    










NUM_THREADS = 5
q = Queue()


def train_model():
	"""
	Download image from img_url in curent directory
	"""
	global q

	while True:
		img_url = q.get()

		res = requests.get(img_url, stream=True)
		filename = f"{img_url.split('/')[-1]}.jpg"

		with open(filename, 'wb') as f:
			for block in res.iter_content(1024):
				f.write(block)
		q.task_done()


if __name__ == '__main__':
    num_epochs = [16,32,64]
    optimizers = ['adam','SGD']
    
    for img_url in images * 5:
        q.put(img_url)

    for t in range(NUM_THREADS):

        worker = Thread(target=download_img)
        worker.daemon = True
        worker.start()

    q.join()