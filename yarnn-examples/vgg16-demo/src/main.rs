use yarnn::native::Native;
use yarnn::optimizers::Adam;
use yarnn_model_vgg16::Vgg16Model;


fn main() {
    let vgg16: Vgg16Model<f32, Native<_>, Adam<_, _>> = Vgg16Model::new(224, 224, 3);

    println!("{}", vgg16);
}
