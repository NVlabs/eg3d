function linkVideos(sliderIndex) {
    videos = document.getElementsByClassName("slider-video-" + sliderIndex)

    for (i = 0; i < videos.length; i++) {
        /* link all the videos to start-stop-seek at the same time */
        linkVideos(videos[i], videos)
            // videos[i].play()
    }

    function linkVideos(video, videoList) {
        video.addEventListener("pause", pauseAll)
        video.addEventListener("play", playAll)

        video.addEventListener("click", toggleState)

        function toggleState(e) {
            if (video.paused) {
                video.play()
            } else {
                video.pause()
            }
        }
        // video.addEventListener("seeked", seekAll)



        function pauseAll(e) {
            for (i = 0; i < videoList.length; i++) {
                videoList[i].pause()
            }
        }

        function playAll(e) {
            for (i = 0; i < videoList.length; i++) {
                videoList[i].play()
            }
        }

        function seekAll(e) {
            e.stopPropagation();
            e.stopImmediatePropagation();
            for (i = 0; i < videoList.length; i++) {
                videoList[i].currentTime = video.currentTime
            }
        }
    }
}

function resetComparisons() {
    videos = document.getElementsByClassName("slider-video")

    for (i = 0; i < videos.length; i++) {
        videos[i].pause()
        videos[i].currentTime = 0
    }
}

// Credit to https://www.w3schools.com/howto/howto_js_image_comparison.asp
function initComparisons(sliderIndex) {
    videos = document.getElementsByClassName("slider-video")
    for (i = 0; i < videos.length; i++) {
        videos[i].removeAttribute("controls")
        videos[i].muted = true
        maxwidth = document.getElementsByClassName("img-comp-container")[0].clientWidth
        videos[i].width = maxwidth;
    }

    var x, i;
    /* Find all elements with an "overlay" class: */
    x = document.getElementsByClassName("img-comp-overlay");
    for (i = 0; i < x.length; i++) {
        /* Once for each "overlay" element:
        pass the "overlay" element as a parameter when executing the compareImages function: */
        compareImages(x[i]);
        console.log(i)
        console.log(x[i])
    }

    function compareImages(img) {
        var slider, img, clicked = 0,
            w, h;
        /* Get the width and height of the img element */
        w = img.offsetWidth;
        h = img.offsetHeight;
        /* Set the width of the img element to 50%: */
        img.style.width = (w / 2) + "px";
        /* Create slider: */
        slider = document.createElement("img");
        slider.setAttribute("class", "img-comp-slider");
        slider.setAttribute("src", "media/graphics/arrow.png");
        /* Insert slider */
        img.parentElement.insertBefore(slider, img);
        /* Position the slider in the middle: */
        slider.style.top = (h / 2) - (slider.offsetHeight / 2) + "px";
        slider.style.left = (w / 2) - (slider.offsetWidth / 2) + "px";
        /* Execute a function when the mouse button is pressed: */
        slider.addEventListener("mousedown", slideReady);
        /* And another function when the mouse button is released: */
        window.addEventListener("mouseup", slideFinish);
        /* Or touched (for touch screens: */
        slider.addEventListener("touchstart", slideReady);
        /* And released (for touch screens: */
        window.addEventListener("touchend", slideFinish);

        function slideReady(e) {
            /* Prevent any other actions that may occur when moving over the image: */
            e.preventDefault();
            /* The slider is now clicked and ready to move: */
            clicked = 1;
            /* Execute a function when the slider is moved: */
            window.addEventListener("mousemove", slideMove);
            window.addEventListener("touchmove", slideMove);
        }

        function slideFinish() {
            /* The slider is no longer clicked: */
            clicked = 0;
        }

        function slideMove(e) {
            var pos;
            /* If the slider is no longer clicked, exit this function: */
            if (clicked == 0) return false;
            /* Get the cursor's x position: */
            pos = getCursorPos(e)
                /* Prevent the slider from being positioned outside the image: */
            if (pos < 0) pos = 0;
            if (pos > w) pos = w;
            /* Execute a function that will resize the overlay image according to the cursor: */
            slide(pos);
        }

        function getCursorPos(e) {
            var a, x = 0;
            e = (e.changedTouches) ? e.changedTouches[0] : e;
            /* Get the x positions of the image: */
            a = img.getBoundingClientRect();
            /* Calculate the cursor's x coordinate, relative to the image: */
            x = e.pageX - a.left;
            /* Consider any page scrolling: */
            x = x - window.pageXOffset;
            return x;
        }

        function slide(x) {
            /* Resize the image: */
            img.style.width = x + "px";
            /* Position the slider: */
            slider.style.left = img.offsetWidth - (slider.offsetWidth / 2) + "px";
        }
    }
}