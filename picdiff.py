from picdiff_lib.iimg import *

import sys


def test_text():
    uut = Img(
        img_path="../Pics/AP37-02.jpg",
        logo_path="../Standards/AP23BNA-logo.png",
    )
    uut.cut_text_area(method="cv")
    data_uut = max_conf_text(uut.text_img.copy(), uut.name)
    with open("uut.json", "w") as f:
        json.dump(data_uut, f, indent=4)

    # stand = Img(
    #     img_path="../Pics/AP27BNA-03.jpg",
    #     logo_path="../Standards/AP23BNA-logo.png",
    # )
    # stand.cut_text_area(method="cv")
    # data_std = max_conf_text(stand.text_img.copy(), stand.name)


def test_size():
    uuts = list_path_images("Pics")
    imgs = []
    for uut in uuts:
        img = Img(uut, logo_path="Standards/AP23BNA-logo.png")
        imgs.append(img)
        img.cut_text_area(method="cv")


def picdiff(uut_path, std_path):
    uut = Img(img_path=uut_path)
    std = Img(img_path=std_path)

    diff_image, diff_data, htmldiff, img_ps = compare_text(uut, std)

    diff_img_path = f"{uut.name}-diff.jpg"
    cv.imwrite(diff_img_path, diff_image)
    with open(f"{uut.name}-diff.json", "w") as f:
        json.dump(diff_data, f, indent=4)
    generate_html(diff_img_path, htmldiff, uut.name, img_ps, output_dir=".")

    print(
        f"""======= PicDiff Result =======
Diff image saved to result/{diff_img_path}.
Diff data saved to result/{uut.name}-diff.json.
HTML report saved to result/{uut.name}-diff-report.html."""
    )


if __name__ == "__main__":
    if not os.path.exists("result"):
        os.mkdir("result")
    os.chdir("result")
    if len(sys.argv) < 3:
        print(
            f"""Usage: {sys.argv[0]} <uut_path> <std_path>
                <uut_path> can be a directory or a single image file.
                <std_path> should be a single image file."""
        )
        sys.exit(1)

    uut_path = os.path.abspath(os.path.join("..", sys.argv[1]))
    std_path = os.path.abspath(os.path.join("..", sys.argv[2]))
    if not os.path.exists(uut_path):
        print(f"Error: {uut_path} not found.")
        sys.exit(1)
    if not os.path.exists(std_path):
        print(f"Error: {std_path} not found.")
        sys.exit(1)
    if os.path.isfile(std_path):
        if os.path.isdir(uut_path):
            uuts = list_path_images(uut_path)
            for uut in uuts:
                picdiff(uut, std_path)
        elif os.path.isfile(uut_path):
            picdiff(uut_path, std_path)
    else:
        print(f"Error: {std_path} is not a file.")
        sys.exit(1)
