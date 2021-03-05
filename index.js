// A webcam iterator that generates Tensors from the images from the webcam.
let webcam;

async function Trigger() {
	img = await facedetect();
	await predImage(img);
}	

async function setup(){
	const webcamConfig = {facingMode: 'user', resizeWidth: 224, resizeHeight: 224};
	const cam = await tf.data.webcam(player, webcamConfig);
	const imgre = await cam.capture();
}

async function predImage(img) {
	console.log('loading model...');
	console.log('app');
	const model = await tf.loadGraphModel('model/model.json');
    console.log('Sucessfully loaded model');
	const result = await model.predict(img);
	days = tf.cast(result, 'int32');
	const tensorData = days.dataSync();
	document.getElementById("demo").innerHTML = tensorData[0];
	img.dispose();
}

async function facedetect() {
	var Size = 224;
	const webcamConfig = {facingMode: 'user', resizeWidth: 224, resizeHeight: 224};
	const cam = await tf.data.webcam(player, webcamConfig);	
	const img = await cam.capture();
	const processedImg =
		tf.tidy(() => img.toFloat());
	
  // Load the model.
  const model = await blazeface.load();
	console.log("Blazeface loaded");
  // Pass in an image or video to the model. The model returns an array of
  // bounding boxes, probabilities, and landmarks, one for each detected face.

  const returnTensors = false; // Pass in `true` to get tensors back, rather than values.
  //const predictions = await model.estimateFaces(document.querySelector("img"), returnTensors);
	const predictions = await model.estimateFaces(processedImg, returnTensors);
  if (predictions.length > 0) {
    for (let i = 0; i < predictions.length; i++) {
      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;  
	  const size = [end[0] - start[0], end[1] - start[1]];
	
	  var increase = parseInt(((predictions[0].topLeft[1] + predictions[0].bottomRight[1])*0.05)/2);
	  //const increase = predictions[0].topLeft[1]; //[predictions[0].topLeft[1] + predictions[0].bottomRight[1];
	  var y1 = (predictions[0].topLeft[1]);
	  var x1 = (predictions[0].topLeft[0]-increase);
	  var y2 = (predictions[0].bottomRight[1]+increase*2);
	  var x2 = (predictions[0].bottomRight[0]+increase);
	  
	    if(y2 > Size){
		console.log("Out of bounds");
		x1 += increase;
		x2 -= increase; 
		y2 = Size;
		x2 = (y2-y1)+x1;
		}
		if(x2 > Size){
		console.log("Out of bounds");
		x1 += increase;
		x2 -= increase; 
		x2 = Size;
		y2 = (x2-x1)+y1;
		}
	
      // Render a rectangle over each detected face.
		canvas = document.getElementById('output');
		canvas.width = 224;
		canvas.height = 224;
		ctx = canvas.getContext('2d');
		ctx.drawImage(player,0,0);
		ctx.beginPath();
		ctx.lineWidth = "3";
		ctx.strokeStyle = "red";
		//ctx.rect(start[0], start[1], size[0], size[1]);
		ctx.rect(x1,y1,x2-x1,y2-y1)
		ctx.stroke();
    }
  }
  
  const Imgdim =
		tf.tidy(() => img.expandDims(0));
  console.log([y1, x1, y2, x2]);
  const imgre = tf.image.cropAndResize(Imgdim,[[  y1/Size, x1/Size, y2/Size, x2/Size]],[0], [224, 224]);
  
  /* For Testing 
  const processedImg2 =
		tf.tidy(() => imgre.squeeze().toInt());
  const canvasother = document.getElementById('another');   
  const abc = tf.browser.toPixels(processedImg2, canvasother);
  console.log(abc); */
  
  return imgre;
}

setup();