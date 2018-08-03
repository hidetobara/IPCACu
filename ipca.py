from PIL import Image
import numpy
from sklearn.decomposition import IncrementalPCA


class scikit_ipca:
	axis = 16

	def __init(self):
		pass

	def run(self):
		ipca = IncrementalPCA(n_components=self.axis, batch_size=10)
		for m in range(4):
			arr = [];
			for n in range(300):
				path = "data/%04d.jpg" % (n + 300 * m)
				img = Image.open(path)
				if img.mode != "RGB":
					img = img.convert("RGB")
				data = numpy.asarray(img, dtype=numpy.float64) / 255.0;
				arr.append(data.flatten())
			ipca.partial_fit(arr)
		self.save("correct/00.png", ipca.mean_);
		for a in range(self.axis):
			self.save("correct/%02d.png" % (a+1), ipca.components_[a])

	def save(self, path, arr):
		vAve = numpy.average(arr)
		vStd = numpy.std(arr)
		vMin = vAve - vStd * 2.0
		vMax = vAve + vStd * 2.0
		#print(str([vAve, vStd, vMin, vMax]))
		arr = (arr - vMin) / (vMax - vMin) * 255.0
		img = arr.reshape(32, 32, 3)
		Image.fromarray(numpy.uint8(img)).save(path)
		

i = scikit_ipca()
i.run()
