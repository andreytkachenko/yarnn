#![feature(trait_alias)]

use yarnn::layers::*;
use yarnn::layer::*;
use yarnn::model;

model! {
    Vgg16Model (h: u32, w: u32, c: u32) {
        input_shape: (c, h, w),
        layers: {
            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 64, kernel: (3, 3) },
            ReLu<N, B>,
            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 64, kernel: (3, 3) },
            ReLu<N, B>,
            MaxPool2d<N, B> { pool: (2, 2) },

            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 128, kernel: (3, 3) },
            ReLu<N, B>,
            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 128, kernel: (3, 3) },
            ReLu<N, B>,
            MaxPool2d<N, B> { pool: (2, 2) },

            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 256, kernel: (3, 3) },
            ReLu<N, B>,
            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 256, kernel: (3, 3) },
            ReLu<N, B>,
            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 256, kernel: (3, 3) },
            ReLu<N, B>,
            MaxPool2d<N, B> { pool: (2, 2) },

            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 512, kernel: (3, 3) },
            ReLu<N, B>,
            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 512, kernel: (3, 3) },
            ReLu<N, B>,
            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 512, kernel: (3, 3) },
            ReLu<N, B>,
            MaxPool2d<N, B> { pool: (2, 2) },

            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 512, kernel: (3, 3) },
            ReLu<N, B>,
            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 512, kernel: (3, 3) },
            ReLu<N, B>,
            ZeroPadding2d<N, B> { paddings: (1, 1) },
            Conv2d<N, B, O> { filters: 512, kernel: (3, 3) },
            ReLu<N, B>,
            MaxPool2d<N, B> { pool: (2, 2) },

            Flatten<N, B>,
            Linear<N, B, O> { units: 4096 },
            ReLu<N, B>,
            // Dropout { 0.5 }, // TODO

            Linear<N, B, O> { units: 4096 },
            ReLu<N, B>,
            // Dropout { 0.5 }, // TODO

            Linear<N, B, O> { units: 1000 },
            Softmax<N, B>
        }
    }
}
