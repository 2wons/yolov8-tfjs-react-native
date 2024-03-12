import labels from "../labels.json";

/**
 * Render prediction boxes
 * @param {Expo2DContext} ctx Expo context
 * @param {number} threshold threshold number
 * @param {Array} boxes_data boxes array
 * @param {Array} scores_data scores array
 * @param {Array} classes_data class array
 * @param {Array[Number]} ratios boxes ratio [xRatio, yRatio]
 */
export const renderBoxes = async (
    ctx,
    threshold,
    boxes,
    scores,
    classes,
    ratios,
    flipX = true
) => {
    ctx.clearRect(0, 0, 640, 640); // clean canvas

    let colors = new Colors();

    // render each box
    for (let i = 0; i < scores.length; ++i) {
        if ( !(scores[i] > threshold)) {
            continue;
        }
        const label = labels[classes[i]];
        const color = colors.get(classes[i]);
        const score = (scores[i] * 100).toFixed(1);

        let [y1, x1, y2, x2] = boxes.slice(i * 4, (i + 1) * 4);
        // scale the boxes with image ratio
        x1 *= ratios[0];
        x2 *= ratios[0];
        y1 *= ratios[1];
        y2 *= ratios[1];
        const width = x2 - x1;
        const height = y2 - y1;

        // draw box
        ctx.fillStyle = Colors.hexToRgba(color, 0.2);
        ctx.fillRect(x1, y1, width, height);

        console.log("done drawing")
    }
    ctx.flush();
}

class Colors {
    // ultralytics color palette https://ultralytics.com/
    constructor() {
      this.palette = [
        "#FF3838",
        "#FF9D97",
        "#FF701F",
        "#FFB21D",
        "#CFD231",
        "#48F90A",
        "#92CC17",
        "#3DDB86",
        "#1A9334",
        "#00D4BB",
        "#2C99A8",
        "#00C2FF",
        "#344593",
        "#6473FF",
        "#0018EC",
        "#8438FF",
        "#520085",
        "#CB38FF",
        "#FF95C8",
        "#FF37C7",
      ];
      this.n = this.palette.length;
    }
  
    get = (i) => this.palette[Math.floor(i) % this.n];
  
    static hexToRgba = (hex, alpha) => {
      var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result
        ? `rgba(${[parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)].join(
            ", "
          )}, ${alpha})`
        : null;
    };
  }