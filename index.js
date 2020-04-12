let net = null;


 function showFiles() {
    // An empty img element
    let demoImage = document.getElementById('idImage');
    // read the file from the user
    let file = document.querySelector('input[type=file]').files[0];
    const reader = new FileReader();
    reader.onload = function (event) {
        demoImage.src = reader.result;
    }
    reader.readAsDataURL(file);
    app();
}  

async function app(){
	console.log('loading model...');
	// tensorflow.js 1.0.0
	//const MODEL_URL = '/model/model.json'
	// https://js.tensorflow.org/api/1.0.0/#loadGraphModel
	const model = await tf.loadLayersModel('model/model.json');
    	console.log('Sucessfully loaded model');
    	await predict();
}


async function predict(){
    img_ = document.getElementById('idImage');
    if (img_.src != ""){
        const result = await model.predict(img_);
        console.log(result);
    }
}

app();
