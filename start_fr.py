import sys
from PyQt4 import QtGui, QtCore
import cv2
import numpy
import os, time
import re
import os.path
from glob import glob

fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'database'
fn_train = 'trainer.yml'
(im_width, im_height) = (300, 300)
faceCascade = cv2.CascadeClassifier(fn_haar)
recognizer = cv2.createLBPHFaceRecognizer()
TRAINING = 'training'
TESTING = 'testing'


def proba():

	#recognizer = cv2.createLBPHFaceRecognizer()
	# Part 1: Create fisherRecognizer
	print('Training...')
	# Create a list of images and a list of corresponding names
	os.remove(fn_train)
	(images, lables, names, id) = ([], [], {}, 0)
	for (subdirs, dirs, files) in os.walk(fn_dir):

	    for subdir in dirs:

	        names[id] = subdir
	        subjectpath = os.path.join(fn_dir, subdir)
	        for filename in os.listdir(subjectpath):
	            path = subjectpath + '/' + filename
	            lable = id
		    images.append(cv2.imread(path, 0))
	            lables.append(int(lable))
	        id += 1

	(images, lables) = [numpy.array(lis) for lis in [images, lables]]

	print(str(numpy.array(images)))
	recognizer.train(images, numpy.array(lables))
	recognizer.save(fn_train)

def correctlyName(nameUser):
	print("Provera imena: "+str(nameUser))

	(images, lables, names, id) = ([], [], {}, 0)
	for (subdirs, dirs, files) in os.walk(fn_dir):
	    for subdir in dirs:
	    	print("Ime je: "+str(subdir))
	        if(subdir == nameUser):
	        	return False
	        else:
	        	return True

def addUser(nameUser):
	count = 0;
	size = 5
	print("Upao ovde da snimi glavu: " + str(nameUser))
	path = os.path.join(fn_dir, str(nameUser))
	if not os.path.isdir(path):
	    os.mkdir(path)
	haar_cascade = cv2.CascadeClassifier(fn_haar)
	webcam = cv2.VideoCapture(0)


	print "-----------------------Taking pictures----------------------"
	print "--------------------Give some expressions---------------------"


	while count < 100:
	    (rval, im) = webcam.read()
	    #im = cv2.flip(im, 1, 0)
	    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	    mini = cv2.resize(gray, (gray.shape[1] / 5, gray.shape[0] / 5))
	    faces = haar_cascade.detectMultiScale(mini)
	    faces = sorted(faces, key=lambda x: x[2])
	    if faces:
	        face_i = faces[0]
	        (x, y, w, h) = [v * 5 for v in face_i]
	        face = gray[y:y + h, x:x + w]
	        face_resize = cv2.resize(face, (im_width, im_height))
	        pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
	               if n[0]!='.' ]+[0])[-1] + 1
	        cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
	        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
	        cv2.putText(im, str(nameUser), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
		#time.sleep(0.38)        
		count += 1
	   
	    	
	    cv2.imshow('OpenCV', im)
	    key = cv2.waitKey(10)
	    if key == 27:
	        break
	print str(count) + " images taken and saved to " + str(nameUser) +" folder in database "
	cv2.destroyWindow("OpenCV")

def predictionDetectFace(images, lables, image):
	recognizer.train(images, numpy.array(lables))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.25, 6,flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
	while True:
		for (x, y, w, h) in faces:
			cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
			face = gray[y:y + h, x:x + w]
			face_resize = cv2.resize(face, (im_width, im_height))
			prediction = recognizer.predict(face_resize)
			if prediction[1] < 150:
				print(str(prediction[0]) + " : " + str(prediction[1]))
				return 1, 0, 0, True
			if prediction[1] < 250:
				#print("<200: " + str(prediction[1]))
				return 0, 0, 1, False
			else:
				return 0, 1, 0, False

def predkcijaSlike(path, image):
	(images, lables, names, id) = ([], [], {}, 0)

	
	count = 0
	tp = 0
	fp = 0
	tn = 0
	user = ""
	for filename in os.listdir(path):
		lable = 1
		images.append(cv2.imread(path+'/'+filename, 0))
		lables.append(int(lable))

		tpp, fpp, tnn, us = predictionDetectFace(images, lables, image)
		tp += tpp
		tn += tnn
		fp += fpp
		if us:
			userName = path.split('/')
			user = userName[1]
			print(user)
		count += 1
		if 9 < count:
			return tp, tn, fp, user

def searchName():
	(images, lables, names, id) = ([], [], {}, 0)
	for (subdirs, dirs, files) in os.walk(fn_dir):
	    for subdir in dirs:
	        names[id] = subdir
	        subjectpath = os.path.join(fn_dir, subdir)
	        for filename in os.listdir(subjectpath):
	            path = subjectpath + '/' + filename
	            lable = id
		    images.append(cv2.imread(path, 0))
	            lables.append(int(lable)) 
	        id += 1
	return images, lables, names, id


class ControlWindow(QtGui.QWidget):
    def __init__(self, *args):
        super(QtGui.QWidget, self).__init__()

        self.nameUser = ""
        self.image = ""
        self.save = False


        self.start_button = QtGui.QPushButton('Upaliti kameru')
        self.start_button.clicked.connect(self.startCapture)
        
        self.start_img = QtGui.QPushButton('Ucitati sliku')
        self.start_img.clicked.connect(self.startImage)

        self.update_button = QtGui.QPushButton('Start/Stop azuriranje')
        self.update_button.clicked.connect(self.updateUser)

        # ------ Modification ------ #
        self.capture_button = QtGui.QPushButton('Dodati novog korisnika')
        self.capture_button.clicked.connect(self.addNewUser)

        self.training_button = QtGui.QPushButton('Istrenirajte...')
        self.training_button.clicked.connect(self.training)

        self.testtt = QtGui.QPushButton('Tacnost')
        self.testtt.clicked.connect(self.test)
        # ------ Modification ------ #

        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.start_img)
        vbox.addWidget(self.update_button)

        # ------ Modification ------ #
        vbox.addWidget(self.capture_button)
        vbox.addWidget(self.training_button)
        vbox.addWidget(self.testtt)
        # ------ Modification ------ #

        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(100,100,200,200)
        self.show()

        self.recognizerMetod()
        print("recognizer: " + str(recognizer))

    def test(self):
		print("Testiranje validnosti")
		
		fileName = QtGui.QFileDialog.getOpenFileName(self, 'OpenFile')
		imagePath = str(fileName)

		print("Molimo vas sacekajte koji trenutak.......")

		font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
		self.image = cv2.imread(imagePath)

		images, labels, names, id = searchName()
		cout = 0;
		ime = ""
		(full_res, true_positiv, fals_positiv, true_negative) = (0.0, 0.0, 0.0, 0.0)
		while cout < len(names):
			path = fn_dir + "/" + names[cout]
			print("Koliko ispisuje: " + str(path))
			tp, tn, fp, user = predkcijaSlike(path, self.image)
			full_res += (tp + fp + tn)
			true_positiv += tp
			fals_positiv += fp
			true_negative += tn
			if user != "":
				print("Kad je ovde")
				ime = user
			print(ime)
			cout += 1

		precision = ((true_positiv)/(true_positiv + fals_positiv))
		recall = ((true_positiv)/(true_negative + true_positiv))
		f1 = 2*((precision * recall) / (precision + recall))
		print("true: positiv " + str(true_positiv) + "| true negativ: " + str(true_negative) + "| false positiv: " + str(fals_positiv) + "| full_res: " + str(full_res))
		
		print("=============Osova na sclici je " + str(ime) + " ================")
		print('%s - %.2f' % ("Precison", precision * 100.0) + "%")
		print('%s - %.2f' % ("Recall", recall * 100) + "%")
		print('%s - %.2f' % ("F1-score", f1 * 100) + "%")


    def updateUser(self):
    	print("Nadograditi bazu znanja")
    	if self.save:
    		self.save = False
    	else:
    		self.save = True


    def recognizerMetod(self):
		recognizer.load('trainer.yml')
		print(str("Ucitana pamet"))

    def startCapture(self):
    	print("Pokrenuti kameru za prepoznavanje")
    	#viewCam(self.recognizer);
    	images, labels, names, id = searchName()

    	face_cascade = cv2.CascadeClassifier(fn_haar)
    	webcam = cv2.VideoCapture(0)
    	while True:
    		(_, im) = webcam.read()
    		im = cv2.flip(im, 1, 0)
    		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    		faces = face_cascade.detectMultiScale(gray, 1.25, 6,flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    		for (x,y,w,h) in faces:
    			cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
    			face = gray[y:y + h, x:x + w]
    			face_resize = cv2.resize(face, (im_width, im_height))
    			# Try to recognize the face
    			prediction = recognizer.predict(face_resize)
    			print(str(prediction[0]) + " - " + str(prediction[1]))
    			cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    			if prediction[1] < 60:
    				if prediction[1] > 70:
    					print("ispitati da li je to ta osoba")
    				else:
    					cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    					if self.save:
    						cv2.imshow('Lice', face_resize)
    						nameUser = names[prediction[0]]
    						path = os.path.join(fn_dir, str(nameUser))
    						if not os.path.isdir(path):
    							os.mkdir(path)
    						pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
    								if n[0]!='.' ]+[0])[-1] + 1
    						cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

    			else:
    				cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    		cv2.imshow('OpenCV', im)
    		if cv2.waitKey(27) & 0xFF == ord('q'):
    			break
    		cv2.destroyWindow("Video")

    def startImage(self):
    	print("Pokrenuti prepoznavanje osobe iz slike")

    	fileName = QtGui.QFileDialog.getOpenFileName(self, 'OpenFile')
      	print("fileName: " + str(fileName))
    	imagePath = str(fileName)

    	font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
    	self.image = cv2.imread(imagePath)
    	self.viewImage(self.image)

    def addNewUser(self):
    	print("Dodavanje novog u bazu")
    	odg = True

        self.nameUser, result = QtGui.QInputDialog.getText(self, "Kreiranje imena baze!",
                                            "Unesite ime osobe koju zelite da dodate u sistem")
        if result:

        	correctly = correctlyName(self.nameUser)
        	if correctly:
        		print("Osoba koju zelite da dodate %s!" % self.nameUser)
        		addUser(self.nameUser)
        		print("Dodali ste novog korisnika sada ga treba dodati u yml");
        		proba()
        		print("Dodali smo novog")
        		self.recognizerMetod()
        	else:
        		print("Ovo ime je zauzeto u bazi probajte sa nekim drugim")
        		while odg:
        			self.nameUser, result = QtGui.QInputDialog.getText(self, "Kreiranje imena baze!",
                                            "Ime je zauzeto probajte neko drugo")
			        if result:

			        	correctly = correctlyName(self.nameUser)
			        	if correctly:
			        		print("Osoba koju zelite da dodate %s!" % self.nameUser)
			        		odg = False;
			        	else:
			        		print("Ovo ime je zauzeto u bazi probajte sa nekim drugim")
			        		odg = True;
			        	addUser(self.nameUser)
			        	proba()
			        	self.recognizerMetod()



    def training(self):
    	print("Treniranje neuronske mreze")
    	proba()
    	print("Zavrsio je trening")
    	self.recognizerMetod()


    def viewImage(self, image):
		images, labels, names, id = searchName()

		font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,3,1,0,4)
		#while True:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray, 1.25, 6,flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
		od = True
		while True:
			for (x, y, w, h) in faces:
				cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
				face = gray[y:y + h, x:x + w]
				face_resize = cv2.resize(face, (im_width, im_height))
				prediction = recognizer.predict(face_resize)
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
				print(str(prediction))
				if prediction[1] < 70:
					cv2.cv.PutText(cv2.cv.fromarray(image), '%s - %.0f' % (str(names[prediction[0]]), prediction[1]), (x,y+h),font, 255)
				else:
					cv2.putText(image,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
					
					if od:
						cv2.imshow("add", face_resize)
						nameUser, result = QtGui.QInputDialog.getText(self, "Dodajte novog korisnika! U sistem ce uci vremeno",
		                                        "Unesite ime novog korisnika")
						od = result
						if result:
							path = os.path.join(fn_dir, str(nameUser))
							if not os.path.isdir(path):
								os.mkdir(path)
							count = 0
							while count < 10:
								pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
									if n[0]!='.' ]+[0])[-1] + 1
								cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
								count += 1
							od = False
							self.training()
			cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
			cv2.imshow('Video', image)
			if cv2.waitKey(27) & 0xFF == ord('q'):
				break
		cv2.destroyWindow("Video")




if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())