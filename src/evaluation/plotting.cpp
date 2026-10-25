#include "mvslam/evaluation/plotting.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "mvslam/compat/pyformat.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam::plot {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.evaluation.plotting");
    return instance;
}


cv::Scalar hex_color(const std::string& hex) {
    const auto value = std::stoul(hex.substr(1), nullptr, 16);
    return cv::Scalar(value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF);  // BGR
}

const cv::Scalar kBlack(0, 0, 0);
const cv::Scalar kWhite(255, 255, 255);
const cv::Scalar kGrid(205, 205, 205);

const std::vector<std::string>& palette() {
    static const std::vector<std::string> colors = {kColorRaw, kColorOptimized, kColorLoop, kColorAccent};
    return colors;
}

// --------------------------------------------------------------------------- //
// Data model
// --------------------------------------------------------------------------- //

struct Series {
    bool scatter = false;
    std::vector<double> x;
    std::vector<double> y;
    cv::Scalar color = kBlack;
    double width_pt = 1.5;
    double alpha = 1.0;
    std::string label;  // empty: not in the legend
    char marker = 'o';  // 'o' circle, 's' square
    double marker_area_pt2 = 36.0;
};

struct AxesSpec {
    std::string title;
    std::string xlabel;
    std::string ylabel;
    bool equal_aspect = false;
    std::optional<double> ymin;
    std::optional<double> ymax;
    bool legend = false;
    std::vector<Series> series;
};

/// One CSV per plot: ``series,x,y[,z]``.
class DataCsv {
public:
    explicit DataCsv(bool three_d = false) {
        text_ = three_d ? py::csv_row({"series", "x", "y", "z"}) : py::csv_row({"series", "x", "y"});
    }
    void add(const std::string& series, double x, double y) {
        text_ += py::csv_row({series, py::repr(x), py::repr(y)});
    }
    void add(const std::string& series, double x, double y, double z) {
        text_ += py::csv_row({series, py::repr(x), py::repr(y), py::repr(z)});
    }
    void add_series(const Series& s, const std::string& name) {
        for (std::size_t i = 0; i < s.x.size(); ++i) {
            add(name, s.x[i], s.y[i]);
        }
    }
    void write(const std::filesystem::path& image_path) const {
        std::ofstream out(data_path_for(image_path), std::ios::binary | std::ios::trunc);
        out << text_;
    }

private:
    std::string text_;
};

// --------------------------------------------------------------------------- //
// Rendering primitives
// --------------------------------------------------------------------------- //

class Canvas {
public:
    Canvas(double width_in, double height_in, int dpi)
        : dpi_(dpi), image_(static_cast<int>(std::lround(height_in * dpi)),
                            static_cast<int>(std::lround(width_in * dpi)), CV_8UC3, kWhite) {}

    cv::Mat& image() { return image_; }
    int width() const { return image_.cols; }
    int height() const { return image_.rows; }
    double px(double points) const { return points * dpi_ / 72.0; }

    double font_scale(double size_pt) const { return px(size_pt) / 22.0; }
    int font_thickness(double size_pt) const {
        return std::max(1, static_cast<int>(std::lround(font_scale(size_pt) * 1.4)));
    }
    cv::Size text_size(const std::string& text, double size_pt) const {
        int baseline = 0;
        return cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale(size_pt), font_thickness(size_pt),
                               &baseline);
    }

    /// Draw ``text`` with its left baseline at ``origin``.
    void text(const std::string& text, cv::Point origin, double size_pt, const cv::Scalar& color = kBlack) {
        cv::putText(image_, text, origin, cv::FONT_HERSHEY_SIMPLEX, font_scale(size_pt), color,
                    font_thickness(size_pt), cv::LINE_AA);
    }

    void text_centered(const std::string& text, cv::Point centre_baseline, double size_pt,
                       const cv::Scalar& color = kBlack) {
        const cv::Size size = text_size(text, size_pt);
        this->text(text, cv::Point(centre_baseline.x - size.width / 2, centre_baseline.y), size_pt, color);
    }

    /// Text rotated 90 degrees counter-clockwise, centred on ``centre``.
    void text_vertical(const std::string& text, cv::Point centre, double size_pt) {
        const cv::Size size = text_size(text, size_pt);
        const int pad = font_thickness(size_pt) + 2;
        cv::Mat patch(size.height + 2 * pad + 4, size.width + 2 * pad, CV_8UC3, kWhite);
        cv::putText(patch, text, cv::Point(pad, size.height + pad), cv::FONT_HERSHEY_SIMPLEX, font_scale(size_pt),
                    kBlack, font_thickness(size_pt), cv::LINE_AA);
        cv::Mat rotated;
        cv::rotate(patch, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::Rect target(centre.x - rotated.cols / 2, centre.y - rotated.rows / 2, rotated.cols, rotated.rows);
        target &= cv::Rect(0, 0, image_.cols, image_.rows);
        if (target.area() > 0) {
            cv::Mat dst = image_(target);
            cv::Mat src = rotated(cv::Rect(0, 0, target.width, target.height));
            cv::min(dst, src, dst);  // dark text over a light background
        }
    }

    bool save(const std::filesystem::path& path) const {
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        return cv::imwrite(path.string(), image_);
    }

private:
    int dpi_;
    cv::Mat image_;
};

struct Range {
    double lo = 0.0;
    double hi = 1.0;
    double span() const { return hi - lo; }
};

Range data_range(const std::vector<Series>& series, bool use_x) {
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (const auto& s : series) {
        for (double v : use_x ? s.x : s.y) {
            if (std::isfinite(v)) {
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
        }
    }
    if (!std::isfinite(lo)) {
        return {0.0, 1.0};
    }
    if (hi - lo < 1e-12) {
        const double pad = std::max(0.5, std::fabs(lo) * 0.05);
        return {lo - pad, hi + pad};
    }
    const double margin = 0.05 * (hi - lo);  // matplotlib's default axes margins
    return {lo - margin, hi + margin};
}

double nice_step(double span, int target_ticks) {
    const double raw = span / std::max(1, target_ticks - 1);
    const double magnitude = std::pow(10.0, std::floor(std::log10(raw)));
    const double normalized = raw / magnitude;
    double nice = 10.0;
    if (normalized < 1.5) {
        nice = 1.0;
    } else if (normalized < 2.25) {
        nice = 2.0;
    } else if (normalized < 3.5) {
        nice = 2.5;
    } else if (normalized < 7.5) {
        nice = 5.0;
    }
    return nice * magnitude;
}

std::vector<double> ticks_for(const Range& range, int target_ticks) {
    const double step = nice_step(range.span(), target_ticks);
    std::vector<double> ticks;
    for (double t = std::ceil(range.lo / step) * step; t <= range.hi + step * 1e-9; t += step) {
        ticks.push_back(std::fabs(t) < step * 1e-9 ? 0.0 : t);
    }
    return ticks;
}

std::string tick_label(double value, double step) {
    int decimals = 0;
    if (step < 1.0) {
        decimals = static_cast<int>(std::ceil(-std::log10(step) - 1e-9));
        if (std::fabs(step * std::pow(10.0, decimals) - std::round(step * std::pow(10.0, decimals))) > 1e-6) {
            ++decimals;  // e.g. a 0.25 step needs two decimals
        }
    }
    std::string text = py::sprintf("%.*f", decimals, value);
    if (text == "-0" || (text.rfind("-0.", 0) == 0 && std::stod(text) == 0.0)) {
        text = text.substr(1);
    }
    return text;
}

void dotted_line(cv::Mat& image, cv::Point a, cv::Point b, const cv::Scalar& color, int dot, int gap) {
    const double length = std::hypot(b.x - a.x, b.y - a.y);
    if (length < 1) {
        return;
    }
    const double dx = (b.x - a.x) / length;
    const double dy = (b.y - a.y) / length;
    for (double t = 0; t < length; t += dot + gap) {
        const double end = std::min(length, t + dot);
        cv::line(image, cv::Point(static_cast<int>(a.x + dx * t), static_cast<int>(a.y + dy * t)),
                 cv::Point(static_cast<int>(a.x + dx * end), static_cast<int>(a.y + dy * end)), color, 1);
    }
}

/// Render one set of axes into ``area`` (the full cell including labels).
void render_axes(Canvas& canvas, const AxesSpec& spec, const cv::Rect& area) {
    cv::Mat& image = canvas.image();
    const double tick_pt = 8.0;
    const double label_pt = 9.5;
    const double title_pt = 11.0;
    const int tick_h = canvas.text_size("0", tick_pt).height;
    const int left = area.x + static_cast<int>(canvas.px(60));
    const int right = area.x + area.width - static_cast<int>(canvas.px(12));
    const int top = area.y + static_cast<int>(canvas.px(spec.title.empty() ? 12 : 26));
    const int bottom = area.y + area.height - static_cast<int>(canvas.px(spec.xlabel.empty() ? 24 : 38));
    const cv::Rect plot_area(left, top, std::max(10, right - left), std::max(10, bottom - top));

    Range xr = data_range(spec.series, true);
    Range yr = data_range(spec.series, false);
    if (spec.ymin) {
        yr.lo = *spec.ymin;
    }
    if (spec.ymax) {
        yr.hi = *spec.ymax;
    }
    if (yr.hi <= yr.lo) {
        yr.hi = yr.lo + 1.0;
    }
    if (spec.equal_aspect) {
        // adjustable="datalim": widen the range of the tighter axis so one
        // data unit spans the same number of pixels on both axes.
        const double ux = xr.span() / plot_area.width;
        const double uy = yr.span() / plot_area.height;
        if (ux > uy) {
            const double c = 0.5 * (yr.lo + yr.hi);
            const double half = 0.5 * ux * plot_area.height;
            yr = {c - half, c + half};
        } else {
            const double c = 0.5 * (xr.lo + xr.hi);
            const double half = 0.5 * uy * plot_area.width;
            xr = {c - half, c + half};
        }
    }
    auto to_px = [&](double x, double y) {
        const double u = (x - xr.lo) / xr.span();
        const double v = (y - yr.lo) / yr.span();
        return cv::Point2d(plot_area.x + u * plot_area.width, plot_area.y + (1.0 - v) * plot_area.height);
    };

    // Grid and ticks.
    const auto xticks = ticks_for(xr, 7);
    const auto yticks = ticks_for(yr, 6);
    const double xstep = xticks.size() > 1 ? xticks[1] - xticks[0] : 1.0;
    const double ystep = yticks.size() > 1 ? yticks[1] - yticks[0] : 1.0;
    for (double t : xticks) {
        const int x = static_cast<int>(std::lround(to_px(t, yr.lo).x));
        dotted_line(image, cv::Point(x, plot_area.y), cv::Point(x, plot_area.y + plot_area.height), kGrid, 2, 3);
        cv::line(image, cv::Point(x, plot_area.y + plot_area.height),
                 cv::Point(x, plot_area.y + plot_area.height + static_cast<int>(canvas.px(3))), kBlack, 1);
        canvas.text_centered(tick_label(t, xstep),
                             cv::Point(x, plot_area.y + plot_area.height + static_cast<int>(canvas.px(5)) + tick_h),
                             tick_pt);
    }
    for (double t : yticks) {
        const int y = static_cast<int>(std::lround(to_px(xr.lo, t).y));
        dotted_line(image, cv::Point(plot_area.x, y), cv::Point(plot_area.x + plot_area.width, y), kGrid, 2, 3);
        cv::line(image, cv::Point(plot_area.x - static_cast<int>(canvas.px(3)), y), cv::Point(plot_area.x, y), kBlack, 1);
        const std::string label = tick_label(t, ystep);
        const cv::Size size = canvas.text_size(label, tick_pt);
        canvas.text(label, cv::Point(plot_area.x - static_cast<int>(canvas.px(5)) - size.width, y + size.height / 2),
                    tick_pt);
    }

    // Data, clipped to the plot area.
    cv::Mat clip = image(plot_area);
    for (const auto& s : spec.series) {
        cv::Mat target = s.alpha < 1.0 ? clip.clone() : clip;
        const int shift = 4;
        const double scale = 1 << shift;
        if (!s.scatter) {
            std::vector<cv::Point> points;
            for (std::size_t i = 0; i < s.x.size(); ++i) {
                if (!std::isfinite(s.x[i]) || !std::isfinite(s.y[i])) {
                    if (points.size() > 1) {
                        cv::polylines(target, points, false, s.color,
                                      std::max(1, static_cast<int>(std::lround(canvas.px(s.width_pt)))),
                                      cv::LINE_AA, shift);
                    }
                    points.clear();
                    continue;
                }
                const cv::Point2d p = to_px(s.x[i], s.y[i]) - cv::Point2d(plot_area.x, plot_area.y);
                points.emplace_back(static_cast<int>(std::lround(p.x * scale)), static_cast<int>(std::lround(p.y * scale)));
            }
            if (points.size() > 1) {
                cv::polylines(target, points, false, s.color,
                              std::max(1, static_cast<int>(std::lround(canvas.px(s.width_pt)))), cv::LINE_AA, shift);
            } else if (points.size() == 1) {
                cv::circle(target, points[0], static_cast<int>(canvas.px(s.width_pt) * scale), s.color, -1,
                           cv::LINE_AA, shift);
            }
        } else {
            const double radius = canvas.px(std::sqrt(s.marker_area_pt2) / 2.0);
            for (std::size_t i = 0; i < s.x.size(); ++i) {
                if (!std::isfinite(s.x[i]) || !std::isfinite(s.y[i])) {
                    continue;
                }
                const cv::Point2d p = to_px(s.x[i], s.y[i]) - cv::Point2d(plot_area.x, plot_area.y);
                if (s.marker == 's') {
                    cv::rectangle(target, cv::Point(static_cast<int>(p.x - radius), static_cast<int>(p.y - radius)),
                                  cv::Point(static_cast<int>(p.x + radius), static_cast<int>(p.y + radius)), s.color,
                                  -1, cv::LINE_AA);
                } else {
                    cv::circle(target, cv::Point(static_cast<int>(std::lround(p.x * scale)),
                                                 static_cast<int>(std::lround(p.y * scale))),
                               static_cast<int>(std::lround(radius * scale)), s.color, -1, cv::LINE_AA, shift);
                }
            }
        }
        if (s.alpha < 1.0) {
            cv::addWeighted(target, s.alpha, clip, 1.0 - s.alpha, 0.0, clip);
        }
    }
    cv::rectangle(image, plot_area, kBlack, 1);

    // Labels and title.
    if (!spec.xlabel.empty()) {
        canvas.text_centered(spec.xlabel,
                             cv::Point(plot_area.x + plot_area.width / 2, area.y + area.height - static_cast<int>(canvas.px(8))),
                             label_pt);
    }
    if (!spec.ylabel.empty()) {
        canvas.text_vertical(spec.ylabel, cv::Point(area.x + static_cast<int>(canvas.px(12)), plot_area.y + plot_area.height / 2),
                             label_pt);
    }
    if (!spec.title.empty()) {
        canvas.text_centered(spec.title, cv::Point(plot_area.x + plot_area.width / 2, plot_area.y - static_cast<int>(canvas.px(8))),
                             title_pt);
    }

    // Legend in the least crowded corner (matplotlib's loc="best", simplified).
    if (spec.legend) {
        std::vector<const Series*> entries;
        for (const auto& s : spec.series) {
            if (!s.label.empty()) {
                entries.push_back(&s);
            }
        }
        if (entries.empty()) {
            return;
        }
        const double legend_pt = 8.0;
        const int row_h = static_cast<int>(canvas.px(13));
        const int sample_w = static_cast<int>(canvas.px(20));
        int text_w = 0;
        for (const auto* e : entries) {
            text_w = std::max(text_w, canvas.text_size(e->label, legend_pt).width);
        }
        const int pad = static_cast<int>(canvas.px(5));
        const int box_w = pad * 3 + sample_w + text_w;
        const int box_h = pad * 2 + row_h * static_cast<int>(entries.size());
        const std::vector<cv::Point> corners = {
            {plot_area.x + plot_area.width - box_w - pad, plot_area.y + pad},                    // upper right
            {plot_area.x + pad, plot_area.y + pad},                                               // upper left
            {plot_area.x + pad, plot_area.y + plot_area.height - box_h - pad},                    // lower left
            {plot_area.x + plot_area.width - box_w - pad, plot_area.y + plot_area.height - box_h - pad}};  // lower right
        std::size_t best = 0;
        std::size_t best_count = std::numeric_limits<std::size_t>::max();
        for (std::size_t c = 0; c < corners.size(); ++c) {
            const cv::Rect box(corners[c], cv::Size(box_w, box_h));
            std::size_t count = 0;
            for (const auto& s : spec.series) {
                for (std::size_t i = 0; i < s.x.size(); ++i) {
                    const cv::Point2d p = to_px(s.x[i], s.y[i]);
                    count += box.contains(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y))) ? 1 : 0;
                }
            }
            if (count < best_count) {
                best_count = count;
                best = c;
            }
        }
        const cv::Rect box(corners[best], cv::Size(box_w, box_h));
        cv::Mat region = image(box & cv::Rect(0, 0, image.cols, image.rows));
        cv::Mat white(region.size(), region.type(), kWhite);
        cv::addWeighted(white, 0.9, region, 0.1, 0.0, region);  // framealpha=0.9
        cv::rectangle(image, box, cv::Scalar(180, 180, 180), 1);
        for (std::size_t k = 0; k < entries.size(); ++k) {
            const Series& e = *entries[k];
            const int cy = box.y + pad + row_h * static_cast<int>(k) + row_h / 2;
            const int sx = box.x + pad;
            if (e.scatter) {
                const int r = std::max(2, static_cast<int>(canvas.px(std::sqrt(e.marker_area_pt2) / 2.0)));
                if (e.marker == 's') {
                    cv::rectangle(image, cv::Point(sx + sample_w / 2 - r, cy - r), cv::Point(sx + sample_w / 2 + r, cy + r),
                                  e.color, -1);
                } else {
                    cv::circle(image, cv::Point(sx + sample_w / 2, cy), r, e.color, -1, cv::LINE_AA);
                }
            } else {
                cv::line(image, cv::Point(sx, cy), cv::Point(sx + sample_w, cy), e.color,
                         std::max(1, static_cast<int>(std::lround(canvas.px(e.width_pt)))), cv::LINE_AA);
            }
            canvas.text(e.label, cv::Point(sx + sample_w + pad, cy + canvas.text_size(e.label, legend_pt).height / 2),
                        legend_pt);
        }
    }
}

std::filesystem::path finish(Canvas& canvas, const std::filesystem::path& path, const DataCsv& csv) {
    if (!canvas.save(path)) {
        throw ValueError("could not write plot '" + path.string() +
                         "': the C++ renderer supports raster formats only (png, jpg, bmp)");
    }
    csv.write(path);
    logger().debug("Wrote " + path.string());
    return path;
}

Series line(const Points2& xz, const std::string& color, double width, const std::string& label,
            double alpha = 1.0) {
    Series s;
    for (Eigen::Index i = 0; i < xz.rows(); ++i) {
        s.x.push_back(xz(i, 0));
        s.y.push_back(xz(i, 1));
    }
    s.color = hex_color(color);
    s.width_pt = width;
    s.label = label;
    s.alpha = alpha;
    return s;
}

Series marker(double x, double y, char shape, double area, const std::string& color, const std::string& label) {
    Series s;
    s.scatter = true;
    s.x = {x};
    s.y = {y};
    s.marker = shape;
    s.marker_area_pt2 = area;
    s.color = hex_color(color);
    s.label = label;
    return s;
}

AxesSpec topdown(const std::string& title) {
    AxesSpec spec;
    spec.title = title;
    spec.xlabel = "x [m]";
    spec.ylabel = "z [m]";
    // Equal aspect matters: without it, a drifting trajectory can be squashed
    // into looking accurate.
    spec.equal_aspect = true;
    spec.legend = true;
    return spec;
}

}  // namespace

std::filesystem::path data_path_for(const std::filesystem::path& image_path) {
    return image_path.parent_path() / (image_path.stem().string() + "_data.csv");
}

std::filesystem::path plot_trajectory(const Trajectory& trajectory, const std::filesystem::path& path,
                                      const std::optional<Trajectory>& reference, const std::string& label,
                                      const std::string& title, int dpi, const std::string& color) {
    Canvas canvas(7.5, 7.0, dpi);
    AxesSpec spec = topdown(title);
    DataCsv csv;
    if (reference) {
        spec.series.push_back(line(reference->xz(), kColorGt, 2.0, "Ground truth"));
        csv.add_series(spec.series.back(), "Ground truth");
    }
    spec.series.push_back(line(trajectory.xz(), color, 1.6, label));
    csv.add_series(spec.series.back(), label);
    if (!trajectory.empty()) {
        const Points2 xz = trajectory.xz();
        spec.series.push_back(marker(xz(0, 0), xz(0, 1), 'o', 70, kColorOptimized, "Start"));
        spec.series.push_back(marker(xz(xz.rows() - 1, 0), xz(xz.rows() - 1, 1), 's', 70, kColorAccent, "End"));
        csv.add("Start", xz(0, 0), xz(0, 1));
        csv.add("End", xz(xz.rows() - 1, 0), xz(xz.rows() - 1, 1));
    }
    render_axes(canvas, spec, cv::Rect(0, 0, canvas.width(), canvas.height()));
    return finish(canvas, path, csv);
}

std::filesystem::path plot_trajectory_comparison(const NamedTrajectories& trajectories,
                                                 const std::filesystem::path& path,
                                                 const std::optional<Trajectory>& reference, const std::string& title,
                                                 int dpi) {
    Canvas canvas(8.0, 7.5, dpi);
    AxesSpec spec = topdown(title);
    DataCsv csv;
    if (reference) {
        spec.series.push_back(line(reference->xz(), kColorGt, 2.4, "Ground truth"));
        csv.add_series(spec.series.back(), "Ground truth");
    }
    for (std::size_t i = 0; i < trajectories.size(); ++i) {
        const auto& [label, trajectory] = trajectories[i];
        spec.series.push_back(line(trajectory.xz(), palette()[i % palette().size()], 1.6, label));
        csv.add_series(spec.series.back(), label);
    }
    if (reference && !reference->empty()) {
        const Points2 xz = reference->xz();
        spec.series.push_back(marker(xz(0, 0), xz(0, 1), 'o', 80, kColorOptimized, "Start"));
        csv.add("Start", xz(0, 0), xz(0, 1));
    }
    render_axes(canvas, spec, cv::Rect(0, 0, canvas.width(), canvas.height()));
    return finish(canvas, path, csv);
}

std::filesystem::path plot_trajectory_3d(const NamedTrajectories& trajectories, const std::filesystem::path& path,
                                         const std::optional<Trajectory>& reference, const std::string& title,
                                         int dpi) {
    Canvas canvas(9.0, 7.0, dpi);
    DataCsv csv(true);
    struct Curve {
        std::string label;
        cv::Scalar color;
        double width;
        Points3 points;  // plot coordinates (x, z, -y)
    };
    std::vector<Curve> curves;
    auto to_plot = [](const Trajectory& t) {
        Points3 p = t.positions();
        Points3 out(p.rows(), 3);
        for (Eigen::Index i = 0; i < p.rows(); ++i) {
            out.row(i) << p(i, 0), p(i, 2), -p(i, 1);
        }
        return out;
    };
    if (reference) {
        curves.push_back({"Ground truth", hex_color(kColorGt), 2.0, to_plot(*reference)});
    }
    for (std::size_t i = 0; i < trajectories.size(); ++i) {
        curves.push_back({trajectories[i].first, hex_color(palette()[i % palette().size()]), 1.4,
                          to_plot(trajectories[i].second)});
    }
    // _equalize_3d: a cube around the data so no axis is stretched.
    Eigen::Vector3d lo = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector3d hi = -lo;
    for (const auto& c : curves) {
        for (Eigen::Index i = 0; i < c.points.rows(); ++i) {
            for (int k = 0; k < 3; ++k) {
                if (std::isfinite(c.points(i, k))) {
                    lo(k) = std::min(lo(k), c.points(i, k));
                    hi(k) = std::max(hi(k), c.points(i, k));
                }
            }
            csv.add(c.label, c.points(i, 0), c.points(i, 1), c.points(i, 2));
        }
    }
    if (!lo.allFinite()) {
        lo.setZero();
        hi.setOnes();
    }
    const Eigen::Vector3d centre = 0.5 * (lo + hi);
    const double radius = std::max(1e-3, 0.5 * (hi - lo).maxCoeff());

    // Orthographic view from matplotlib's default angles (azim=-60, elev=30).
    const double az = -60.0 * M_PI / 180.0;
    const double el = 30.0 * M_PI / 180.0;
    const Eigen::Vector3d right(-std::sin(az), std::cos(az), 0.0);
    const Eigen::Vector3d up(-std::sin(el) * std::cos(az), -std::sin(el) * std::sin(az), std::cos(el));
    const int margin = static_cast<int>(canvas.px(50));
    const cv::Rect area(margin, static_cast<int>(canvas.px(40)), canvas.width() - 2 * margin,
                        canvas.height() - static_cast<int>(canvas.px(40)) - margin);
    const double extent = radius * 1.8;  // projected cube half-size, with room
    const double pixels_per_unit = 0.5 * std::min(area.width, area.height) / extent;
    auto project = [&](const Eigen::Vector3d& p) {
        const Eigen::Vector3d q = p - centre;
        return cv::Point2d(area.x + area.width / 2.0 + right.dot(q) * pixels_per_unit,
                           area.y + area.height / 2.0 - up.dot(q) * pixels_per_unit);
    };
    // Cube wireframe with labelled axes.
    cv::Mat& image = canvas.image();
    std::vector<Eigen::Vector3d> corners;
    for (int i = 0; i < 8; ++i) {
        corners.push_back(centre + radius * Eigen::Vector3d(i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1));
    }
    for (int i = 0; i < 8; ++i) {
        for (int bit : {1, 2, 4}) {
            const int j = i ^ bit;
            if (j > i) {
                const cv::Point2d a = project(corners[static_cast<std::size_t>(i)]);
                const cv::Point2d b = project(corners[static_cast<std::size_t>(j)]);
                dotted_line(image, cv::Point(static_cast<int>(a.x), static_cast<int>(a.y)),
                            cv::Point(static_cast<int>(b.x), static_cast<int>(b.y)), kGrid, 3, 3);
            }
        }
    }
    const std::vector<std::pair<int, std::string>> axes = {{0, "x [m]"}, {1, "z [m]"}, {2, "height [m]"}};
    for (const auto& [k, label] : axes) {
        Eigen::Vector3d a = centre - Eigen::Vector3d::Constant(radius);
        Eigen::Vector3d b = a;
        b(k) += 2 * radius;
        const cv::Point2d pa = project(a);
        const cv::Point2d pb = project(b);
        cv::line(image, cv::Point(static_cast<int>(pa.x), static_cast<int>(pa.y)),
                 cv::Point(static_cast<int>(pb.x), static_cast<int>(pb.y)), kBlack, 1, cv::LINE_AA);
        canvas.text(label + " (" + py::format(a(k), ".0f") + ".." + py::format(b(k), ".0f") + ")",
                    cv::Point(static_cast<int>(pb.x) + 4, static_cast<int>(pb.y) + 4), 8.0);
    }
    for (const auto& c : curves) {
        std::vector<cv::Point> points;
        for (Eigen::Index i = 0; i < c.points.rows(); ++i) {
            const cv::Point2d p = project(c.points.row(i).transpose());
            points.emplace_back(static_cast<int>(std::lround(p.x * 16)), static_cast<int>(std::lround(p.y * 16)));
        }
        if (points.size() > 1) {
            cv::polylines(image, points, false, c.color, std::max(1, static_cast<int>(std::lround(canvas.px(c.width)))),
                          cv::LINE_AA, 4);
        }
    }
    canvas.text_centered(title, cv::Point(canvas.width() / 2, static_cast<int>(canvas.px(22))), 11.0);
    // Legend.
    int y = static_cast<int>(canvas.px(40));
    for (const auto& c : curves) {
        const int x = canvas.width() - static_cast<int>(canvas.px(150));
        cv::line(image, cv::Point(x, y), cv::Point(x + static_cast<int>(canvas.px(20)), y), c.color,
                 std::max(1, static_cast<int>(std::lround(canvas.px(c.width)))), cv::LINE_AA);
        canvas.text(c.label, cv::Point(x + static_cast<int>(canvas.px(26)), y + static_cast<int>(canvas.px(3))), 8.0);
        y += static_cast<int>(canvas.px(13));
    }
    return finish(canvas, path, csv);
}

std::filesystem::path plot_loop_closures(const Trajectory& trajectory,
                                         const std::vector<std::pair<std::int64_t, std::int64_t>>& loops,
                                         const std::filesystem::path& path, const std::optional<Trajectory>& reference,
                                         const std::string& title, int dpi) {
    Canvas canvas(8.0, 7.5, dpi);
    AxesSpec spec = topdown(title + " (" + std::to_string(loops.size()) + " accepted)");
    DataCsv csv;
    if (reference) {
        spec.series.push_back(line(reference->xz(), kColorGt, 1.6, "Ground truth", 0.5));
        csv.add_series(spec.series.back(), "Ground truth");
    }
    spec.series.push_back(line(trajectory.xz(), kColorRaw, 1.6, "Estimated"));
    csv.add_series(spec.series.back(), "Estimated");
    const Points2 xz = trajectory.xz();
    const auto n = static_cast<std::int64_t>(xz.rows());
    for (std::size_t k = 0; k < loops.size(); ++k) {
        const auto [i, j] = loops[k];
        if (!(0 <= i && i < n && 0 <= j && j < n)) {
            continue;
        }
        Series edge;
        edge.x = {xz(i, 0), xz(j, 0)};
        edge.y = {xz(i, 1), xz(j, 1)};
        edge.color = hex_color(kColorLoop);
        edge.width_pt = 1.1;
        edge.alpha = 0.85;
        edge.label = k == 0 ? "Loop closure" : "";
        spec.series.push_back(edge);
        Series ends = edge;
        ends.scatter = true;
        ends.alpha = 1.0;
        ends.marker_area_pt2 = 14;
        ends.label.clear();
        spec.series.push_back(ends);
        csv.add_series(edge, "loop_" + std::to_string(k));
    }
    render_axes(canvas, spec, cv::Rect(0, 0, canvas.width(), canvas.height()));
    return finish(canvas, path, csv);
}

std::filesystem::path plot_error_over_time(const std::vector<std::pair<std::string, std::vector<double>>>& errors,
                                           const std::filesystem::path& path,
                                           const std::optional<std::vector<double>>& distances,
                                           const std::string& title, int dpi) {
    Canvas canvas(9.0, 4.5, dpi);
    AxesSpec spec;
    spec.title = title;
    spec.xlabel = distances ? "distance travelled [m]" : "frame index";
    spec.ylabel = "position error [m]";
    spec.ymin = 0.0;
    spec.legend = true;
    DataCsv csv;
    for (std::size_t k = 0; k < errors.size(); ++k) {
        const auto& [label, values] = errors[k];
        if (distances && distances->size() < values.size()) {
            throw ValueError("x and y must have same first dimension");
        }
        Series s;
        for (std::size_t i = 0; i < values.size(); ++i) {
            s.x.push_back(distances ? (*distances)[i] : static_cast<double>(i));
            s.y.push_back(values[i]);
        }
        s.color = hex_color(palette()[k % palette().size()]);
        s.width_pt = 1.3;
        s.label = label;
        spec.series.push_back(s);
        csv.add_series(s, label);
    }
    render_axes(canvas, spec, cv::Rect(0, 0, canvas.width(), canvas.height()));
    return finish(canvas, path, csv);
}

std::filesystem::path plot_diagnostics(const std::vector<FrameDiagnostics>& diagnostics,
                                       const std::filesystem::path& path, const std::string& title, int dpi) {
    if (diagnostics.empty()) {
        throw ValueError("No diagnostics to plot");
    }
    std::vector<double> frames, matches, inliers, ratio, rotation, failures;
    double cumulative = 0.0;
    for (const auto& d : diagnostics) {
        frames.push_back(static_cast<double>(d.index));
        matches.push_back(static_cast<double>(d.n_matches));
        inliers.push_back(static_cast<double>(d.n_ransac_inliers));
        ratio.push_back(d.ransac_inlier_ratio * 100.0);
        rotation.push_back(d.rotation_deg);
        cumulative += d.succeeded() ? 0.0 : 1.0;
        failures.push_back(cumulative);
    }
    auto series = [&](const std::vector<double>& y, const std::string& color, double width, const std::string& label) {
        Series s;
        s.x = frames;
        s.y = y;
        s.color = hex_color(color);
        s.width_pt = width;
        s.label = label;
        return s;
    };

    Canvas canvas(12.0, 7.0, dpi);
    const int top = static_cast<int>(canvas.px(28));
    const int cell_w = canvas.width() / 2;
    const int cell_h = (canvas.height() - top) / 2;
    AxesSpec a;
    a.title = "Matches and inliers";
    a.ylabel = "count";
    a.legend = true;
    a.series = {series(matches, kColorLoop, 0.9, "filtered matches"), series(inliers, kColorOptimized, 0.9, "epipolar inliers")};
    AxesSpec b;
    b.title = "RANSAC inlier ratio";
    b.ylabel = "inlier ratio [%]";
    b.ymin = 0.0;
    b.ymax = 100.0;
    b.series = {series(ratio, kColorRaw, 0.9, "")};
    AxesSpec c;
    c.title = "Per-frame rotation magnitude";
    c.ylabel = "rotation [deg]";
    c.xlabel = "frame";
    c.series = {series(rotation, kColorLoop, 0.9, "")};
    AxesSpec d;
    d.title = "Failed transitions (" + py::format(static_cast<std::int64_t>(cumulative), "") + " of " +
              std::to_string(diagnostics.size()) + ")";
    d.ylabel = "cumulative failures";
    d.xlabel = "frame";
    d.series = {series(failures, kColorRaw, 1.2, "")};
    render_axes(canvas, a, cv::Rect(0, top, cell_w, cell_h));
    render_axes(canvas, b, cv::Rect(cell_w, top, cell_w, cell_h));
    render_axes(canvas, c, cv::Rect(0, top + cell_h, cell_w, cell_h));
    render_axes(canvas, d, cv::Rect(cell_w, top + cell_h, cell_w, cell_h));
    canvas.text_centered(title, cv::Point(canvas.width() / 2, static_cast<int>(canvas.px(18))), 12.0);

    DataCsv csv;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        csv.add("filtered_matches", frames[i], matches[i]);
    }
    for (std::size_t i = 0; i < frames.size(); ++i) {
        csv.add("epipolar_inliers", frames[i], inliers[i]);
    }
    for (std::size_t i = 0; i < frames.size(); ++i) {
        csv.add("inlier_ratio_pct", frames[i], ratio[i]);
    }
    for (std::size_t i = 0; i < frames.size(); ++i) {
        csv.add("rotation_deg", frames[i], rotation[i]);
    }
    for (std::size_t i = 0; i < frames.size(); ++i) {
        csv.add("cumulative_failures", frames[i], failures[i]);
    }
    return finish(canvas, path, csv);
}

}  // namespace mvslam::plot
