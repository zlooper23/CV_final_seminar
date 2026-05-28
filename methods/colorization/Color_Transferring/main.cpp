#include "include/image.h"
#include "include/image_qt.h"
#include <stdio.h>
#include <QImage>

int main(int argc, char *argv[])
{
    // 1. Define your source/exemplar image (the colored image you are copying from)
    // MAKE SURE TO CHANGE THIS PATH TO YOUR ACTUAL COLORED REFERENCE IMAGE
    const char* source_path = "C:/Users/noelt/Desktop/UPF/CV_Seminar_Project/data/frames/video2/frame_0001.png";

    QImage sourceQImg(source_path);
    if(sourceQImg.isNull()) {
        printf("Error: Could not load the exemplar image. Check the path!\n");
        return 1;
    }
    struct img_rgb_t *source = QImage_to_img_rgb(&sourceQImg);

    int wnd_ht = 25; // size of local windows
    int wnd_wt = 25; // size of local windows

    #pragma omp parallel for
    // 2. Loop through all your frames (adjust 241 to your exact number of frames)
    for (int i = 1; i <= 344; i++) {
        char target_file[256];
        char out_file[256];

        // 3. Format the file paths (the %03d makes it frame_001, frame_002, etc.)
        // CHANGE THESE PATHS TO WHERE YOUR BLACK & WHITE FRAMES ARE AND WHERE YOU WANT TO SAVE THEM
        sprintf(target_file, "C:/Users/noelt/Desktop/UPF/CV_Seminar_Project/methods/colorization/Deep-Learning/Image Colorization Tutorial/data/frames/video/frame_%04d.png", i);
        sprintf(out_file, "C:/Users/noelt/Desktop/UPF/CV_Seminar_Project/outputs/colorization/ExampleBased/video2/frames/color_%04d.png", i);

        printf("Processing Frame %03d...\n", i);

        // Load current B&W frame
        QImage targetQImg(target_file);
        if(targetQImg.isNull()) {
            printf("Could not load %s, skipping...\n", target_file);
            continue; // Skip to the next frame if image is missing
        }

        struct img_rgb_t *target = QImage_to_img_rgb(&targetQImg);

        // Apply color transfer
        struct img_rgb_t *out = transfer_color(target, source, wnd_ht, wnd_wt);

        // Save output
        QImage outImg = img_rgb_to_QImage(out);
        outImg.save(out_file, "PNG");

        // Clean up C-struct memory for this frame to prevent RAM crashes
        img_rgb_destruct(target);
        img_rgb_destruct(out);
    }

    // Clean up source image memory
    img_rgb_destruct(source);

    printf("SUCCESS: All frames processed!\n");
    return 0;
}