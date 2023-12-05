from manim import *
from manim_slides import Slide
from glob import glob

color1 = ManimColor([153, 184, 152])
color2 = ManimColor([254, 206, 168])
color3 = ManimColor([255, 132, 124])
color4 = ManimColor([232,  74,  95])
color5 = ManimColor([ 42,  54,  59])

files = sorted(glob("./output/*.png"))

rng = np.random.default_rng(seed=54252)

class PerlinNoiseSlides(Slide):
    def construct(self):
        overlay = Rectangle(color=color5, height=9, width=16, stroke_width=0).set_opacity(0.9)
        img = ImageMobject(files[-1])
        self.add(img)

        perlin = Text("Perlin", color=color2, font_size=64, )
        noise = Text("Noise", color=color2, font_size=64, ).next_to(perlin, RIGHT)
        title = VMobject()
        title.add(perlin, noise)
        title.move_to(ORIGIN + 0.3 * UP)

        name = Text("- Tapan Mayukh", color=color2, font_size=32).shift(0.5 * DOWN + 3 * RIGHT)
        self.play(FadeIn(overlay))
        self.play(LaggedStart(Write(title), Write(name), lag_ratio=0.4), run_time=3)
        self.next_slide()

        self.play(title.animate.to_corner(UP, buff=0.2).scale(0.7), Uncreate(name), run_time=1.5)
        self.next_slide()

        perlin_ul = Underline(perlin, color=color3)
        noise_ul  = Underline(noise,  color=color3)
        self.play(ShowPassingFlash(perlin_ul), ShowPassingFlash(noise_ul))
        self.next_slide()

        self.play(Create(noise_ul))

        linear_axes = Axes(x_range=[-1.1, 4.1], y_range=[-1.1, 4.1], x_length=5, y_length=5, \
                          axis_config={"include_numbers": False, "include_tip": False, "color": color2})
        labels = linear_axes.get_axis_labels(x_label="x", y_label="y")
        labels.set_color(color2)
        x_vals = np.linspace(-1, 4, 11)
        y_vals = np.linspace(-1, 4, 11)
        linear_plot = linear_axes.plot(lambda x: x, color=color1)

        dots_curr = linear_axes.coords_to_point(x_vals, y_vals)
        dots = VGroup()
        for d0, d1 in zip(dots_curr[0], dots_curr[1]):
            dots.add(Dot(point=(d0, d1, 0.0), fill_color=color4, stroke_width=0, radius=0.05))
        self.play(Create(linear_axes), Create(labels))
        self.play(LaggedStart(Create(linear_plot), Write(dots), lag_ratio=0.5))
        self.next_slide()

        noise = rng.random(size=(11, )) * 2 - 1.0
        y_new_vals = y_vals + noise
        dots_moved = linear_axes.coords_to_point(x_vals, y_new_vals)

        dots_new = VGroup()
        for d0, d1 in zip(dots_moved[0], dots_moved[1]):
            dots_new.add(Dot(point=(d0, d1, 0.0), fill_color=color4, stroke_width=0, radius=0.05))
        
        lines = VGroup()
        for d0, d1 in zip(dots, dots_new):
            lines.add(DashedLine(d0, d1, stroke_width=0.6 * DEFAULT_STROKE_WIDTH))

        self.play(LaggedStart(Transform(dots, dots_new), FadeIn(lines), lag_ratio=0.5))
        self.next_slide()

        ax1 = VGroup(linear_axes, dots, labels, linear_plot, lines)
        self.play(ax1.animate.shift(4 * RIGHT))

        noise_axes1 = Axes(x_range=[1, 6], y_range=[-1, 1], x_length=5, y_length=2, \
                           x_axis_config={"include_numbers": False, "include_tip": False, "color": color2}, \
                            y_axis_config={"include_numbers": False, "include_tip": False, "stroke_width": 0, "include_ticks": False}).shift(-4 * RIGHT)
        
        noise_dots = noise_axes1.coords_to_point(x_vals + 2, noise)
        del dots_new
        dots_new = VGroup()
        for d0, d1 in zip(noise_dots[0], noise_dots[1]):
            dots_new.add(Dot(point=(d0, d1, 0.0), fill_color=color4, stroke_width=0, radius=0.05))
        
        lines_new = VGroup()
        for d0 in dots_new:
            lines_new.add(DashedLine([d0.get_center()[0], 0.0, 0.0], d0, stroke_width=0.6 * DEFAULT_STROKE_WIDTH))
        
        self.play(Create(noise_axes1))
        self.play(ReplacementTransform(dots.copy(), dots_new), ReplacementTransform(lines.copy(), lines_new))
        self.next_slide()

        noise_plot1 = noise_axes1.plot_line_graph(x_vals + 2.0, noise, line_color=color1, add_vertex_dots=False)
        self.play(FadeOut(lines_new))
        self.play(Create(noise_plot1))

        self.next_slide()

        ax2 = VGroup(noise_axes1, dots_new, noise_plot1)
        self.play(ax2.animate.shift(2 * UP))

        noise_axes2 = noise_axes1.copy().shift(4 * DOWN)
        self.play(ReplacementTransform(noise_axes1.copy(), noise_axes2))

        x_vals = np.linspace(1, 6, 101)
        y_vals = rng.random(size=(101, )) * 2 - 1
        noise_plot2 = noise_axes2.plot_line_graph(x_vals, y_vals, line_color=color1, add_vertex_dots=False, stroke_width=0.5*DEFAULT_STROKE_WIDTH)
        self.play(Create(noise_plot2))
        self.next_slide()