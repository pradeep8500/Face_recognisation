import sys
import os
import dlib
import glob
import datetime

predictor_path = 'shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
faces_folder_path = 'test3'
output_folder_path = 'faces'
output_folder_path1 = 'group1'

currentDT = datetime.datetime.now()

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

descriptors = []
images = []
# Now find all the faces and compute 128D face descriptors for each face.
for f in glob.glob(os.path.join(faces_folder_path, "*")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)

        # Compute the 128D vector that describes the face in img identified by
        # shape.  
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        
        descriptors.append(face_descriptor)
        images.append((img, shape))
#print('face_descriptor',descriptors)
# Now let's cluster the faces.  
labels = dlib.chinese_whispers_clustering(descriptors, 0.4)
num_classes = len(set(labels))

for j in range(0, num_classes):
	indices = []
	for i, label in enumerate(labels):
		if label == j:
			indices.append(i)
	if len(indices) >=3 :
		print("Indices of images in the biggest cluster: {}".format(str(indices)))

		# Ensure output directory exists
		if not os.path.isdir(output_folder_path+'/####'+str(j)):
			os.makedirs(output_folder_path+'/####'+str(j))

		if not os.path.isdir(output_folder_path1+'/####'+str(j)):
			os.makedirs(output_folder_path1+'/####'+str(j))
		name_image=currentDT.strftime("%Y-%m-%d %H:%M:%S")
		# Save the extracted faces
		for i, index in enumerate(indices):
			
			img, shape = images[index]
			file_path = os.path.join(output_folder_path+'/####'+str(j), str(name_image)+'-' +str(j)+str(i))
			# The size and padding arguments are optional with default size=150x150 and padding=0.25
			dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)
			file_path1 = os.path.join(output_folder_path1+'/####'+str(j), str(name_image) + '-'+str(j)+str(i)+'.jpg')
			#image path must end with one of [.bmp, .png, .dng, .jpg, .jpeg] for dlib.save_image
			dlib.save_image(img,file_path1)
    
