var extract = document.getElementById("extracted_image");
var mask = document.getElementById("mask_image");
var prediction = document.getElementById("prediction");
var res = document.getElementById("result");

function maskImage(){
   setTimeout(() => {
     mask.style.display = 'block';
}, 3000);
};

function extractedImage(){
   setTimeout(() => {
     extract.style.display = 'block';
}, 3000);

};

function predict(){
prediction.style.display = 'block';

   setTimeout(() => {
      prediction.style.display = 'none';
      res.style.display = 'block';
}, 10000);
};


