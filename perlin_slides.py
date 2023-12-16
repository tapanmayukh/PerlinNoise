from manim import *
from manim_slides import Slide
from glob import glob
from perlin_noise import PerlinNoise

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

        perlin = Text("Perlin", color=color2, font_size=64)
        noise = Text("Noise", color=color2, font_size=64).next_to(perlin, RIGHT)
        title = VMobject()
        title.add(perlin, noise)
        title.move_to(ORIGIN + 0.3 * UP)

        name = Text("- Tapan Mayukh", color=color2, font_size=32).shift(0.5 * DOWN + 3 * RIGHT)
        self.play(FadeIn(overlay))
        self.play(LaggedStart(Write(title), Write(name), lag_ratio=0.4), run_time=3)
        self.next_slide()

        self.play(title.animate.to_edge(UP, buff=0.2).scale(0.7), Uncreate(name), run_time=1.5)
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

        self.play(FadeOut(ax1, ax2, noise_axes2, noise_plot2))
        self.play(Uncreate(noise_ul))
        self.play(Create(perlin_ul))
        self.next_slide()

        ken_img = ImageMobject("./assets/ken_perlin.jpg").to_edge(LEFT, buff=DEFAULT_MOBJECT_TO_EDGE_BUFFER * 1.5)
        ken_name = Text("Prof. Kenneth H. Perlin", font_size=24, color=color3).next_to(ken_img, DOWN)
        ken_desig = Text("Dept. of Comp Sci, NYU", font_size=24, color=color3).next_to(ken_name, DOWN, buff=0.1)

        self.play(FadeIn(ken_img), Write(ken_name), Write(ken_desig))
        self.next_slide()

        vase_img = ImageMobject("./assets/perlin_vase.jpg").to_edge(RIGHT, buff=DEFAULT_MOBJECT_TO_EDGE_BUFFER * 1.5)
        # vase_img.scale(0.5)
        self.play(FadeIn(vase_img))
        self.next_slide()

        oscar_img = ImageMobject("./assets/oscar.png").scale(0.2).shift(RIGHT)
        oscar_name1 = Text("Academy Award for", font_size=24, color=color3)
        oscar_name2 = Text("Techical Achievement (1996)", font_size=24, color=color3)
        oscar_name1.next_to(oscar_img, DOWN)
        oscar_name2.next_to(oscar_name1, DOWN, buff=0.1)
        self.play(FadeIn(oscar_img))
        self.play(Write(oscar_name1), Write(oscar_name2))
        self.next_slide()

        self.play(FadeOut(ken_img), FadeOut(ken_name), FadeOut(ken_desig), FadeOut(vase_img), \
                  FadeOut(oscar_img), FadeOut(oscar_name1), FadeOut(oscar_name2))
        self.play(Uncreate(perlin_ul))
        self.next_slide()

        del dots
        del lines
        dots, lines, beziers = self.perlin_1d_demo()
        self.play(Create(dots))
        self.next_slide()

        self.play(Create(lines))
        self.next_slide()

        cubic_text = Text("Cubic Interpolation: ", color=color3, font_size=28)
        cubic_eq = MathTex(r"a x^3 + b x^2 + c x + d", fill_color=color2, font_size=36)

        cubic = Group(cubic_text, cubic_eq).arrange(RIGHT, buff=0.4 * DEFAULT_MOBJECT_TO_MOBJECT_BUFFER).shift(3 * DOWN)
        cubic_eq.shift(0.05 * UP)

        self.play(LaggedStart(Write(cubic_text), Write(cubic_eq), lag_ratio=0.8))
        self.next_slide()

        self.play(Create(beziers))
        self.next_slide()

        self.play(Uncreate(beziers), Uncreate(lines), Uncreate(dots), Uncreate(cubic_text), Uncreate(cubic_eq))
        self.next_slide()

        self.octaves(2.0, 0.5)
        self.perlin_2d_demo()

        squares = VGroup()
        seeds = []
        for i in range(4):
            seed = int(rng.random() * 100000000)
            seeds.append(seed)
            perlin_2d = PerlinNoise(octaves=5, seed=seed)
            sq = VGroup()
            for y_idx in range(20):
                for x_idx in range(20):
                    x = (x_idx + 0.5) / 20.0
                    y = (y_idx + 0.5) / 20.0
                    col = perlin_2d([x, y]) + 0.5

                    new_sq = Square(0.15, fill_color=ManimColor([col, col, col]), stroke_width=0, fill_opacity=1).move_to((x_idx * 0.15 - 0.825) * RIGHT + (y_idx * 0.15 - 0.825) * UP)
                    sq.add(new_sq)
            squares.add(sq)
        
        squares.arrange(RIGHT)
        self.play(FadeIn(squares))
        
        seed_txts = VGroup()
        for i, seed in enumerate(seeds):
            seed_txt = Tex(f"{seed}", color=color1, font_size=28).next_to(squares[i], UP, aligned_edge=LEFT)
            seed_txts.add(seed_txt)

        self.play(Write(seed_txts))
        self.next_slide()

        self.play(FadeOut(squares), Uncreate(seed_txts))
        self.next_slide()

        self.perlin_octaves(2.0, 0.5)

        bungle = ImageMobject("./assets/bungle_bungle_range.jpg").scale(0.3)
        bungle.to_edge(LEFT).shift(1.5 * UP)
        bungle_name = Text("Bungle Bungle Range, Australia", font_size=24, color=color1).next_to(bungle, DOWN)
        bungle_art = ImageMobject("./assets/output_2.png").scale(0.3).next_to(bungle_name, DOWN, buff=2.5 * DEFAULT_MOBJECT_TO_MOBJECT_BUFFER)

        tree = ImageMobject("./assets/tree-rings.jpg").scale(0.7).to_edge(RIGHT).shift(1.5 * UP)
        tree_name = Text("Tree Rings", font_size=24, color=color1).next_to(tree, DOWN)
        tree_art = ImageMobject("./assets/output.png").scale(0.7).next_to(tree_name, DOWN, buff=2.5 * DEFAULT_MOBJECT_TO_MOBJECT_BUFFER)

        self.play(FadeIn(bungle), FadeIn(tree))
        self.next_slide()
        self.play(Write(bungle_name), Write(tree_name))
        self.next_slide()

        self.play(FadeIn(bungle_art), FadeIn(tree_art))
        self.next_slide()

        self.play(FadeOut(bungle), FadeOut(bungle_art), FadeOut(tree), FadeOut(tree_art), Uncreate(tree_name), Uncreate(bungle_name))

        where = Text("Where is it used?", font_size=28, color=color1).to_edge(LEFT, buff=2 * DEFAULT_MOBJECT_TO_EDGE_BUFFER).shift(2.5 * UP)
        self.play(Write(where))
        self.next_slide()

        blist = BulletedList("Terrain Generation, e.g. Minecraft, Sid Meier's Civilization, Worldbox, etc.",
                             "Procedural Texture Generation and Computer-generated image",
                             "Flow-field and boid-clustering simulations", buff=2*MED_LARGE_BUFF, color=color2, font_size=28)
        self.play(Write(blist))
        self.next_slide()

        self.play(Uncreate(where), Uncreate(blist))

        where = Text("Okay, but what about Astrophysics?", font_size=28, color=color1).to_edge(LEFT, buff=2 * DEFAULT_MOBJECT_TO_EDGE_BUFFER).shift(2.5 * UP)
        self.play(Write(where))
        self.next_slide()

        blist = BulletedList("Generating naturalistic velocity field initial conditions",
                             "Morphological analysis of asteroid surfaces and their collision history [1]",
                             "Ray-Tracing Galaxies in simulations [2]",
                             buff=2*MED_LARGE_BUFF, color=color2, font_size=28)
        cite2 = Tex("[2] Groenboom and Dahle 2014", font_size=20, color=color4).to_corner(DR)
        cite1 = Tex("[1] Li et. al 2022", font_size=20, color=color4).next_to(cite2, UP, aligned_edge=LEFT, buff=0.5*DEFAULT_MOBJECT_TO_MOBJECT_BUFFER)
        self.play(Write(blist), Write(cite2), Write(cite1))
        self.next_slide()

        self.play(Uncreate(where), Uncreate(blist), Uncreate(cite2), Uncreate(cite1))

        where = Text("Drawbacks and Improvements", font_size=28, color=color1).to_edge(LEFT, buff=2 * DEFAULT_MOBJECT_TO_EDGE_BUFFER).shift(2.5 * UP)
        self.play(Write(where))
        self.next_slide()

        blist = BulletedList("Directional artifacts due to use of squares",
                             "Slow computation of gradients",
                             buff=1.5*MED_LARGE_BUFF, color=color2, font_size=28).next_to(where, DOWN)
        blist2 = BulletedList("Restriction of gradient direction to the axes",
                             "Uses of triangular/tetrahedral blocks instead of square, i.e., Simplex Noise [1]",
                             buff=1.5*MED_LARGE_BUFF, color=color3, font_size=28).next_to(blist, DOWN, aligned_edge=LEFT, buff=1.5*MED_LARGE_BUFF)
        
        cite = Tex("[1] Perlin 2002", font_size=20, color=color4).to_corner(DR)
        self.play(Write(blist), Write(blist2), Write(cite))
        self.next_slide()

        self.play(Uncreate(where), Uncreate(blist), Uncreate(blist2), Uncreate(title), Uncreate(cite))
        self.play(Write(Text("Questions?", color=color3)))
        self.next_slide()


    def perlin_1d_demo(self):
        dots = VGroup()
        for i in range(-6, 7, 6):
            dots.add(Dot(i * RIGHT, DEFAULT_DOT_RADIUS, color=color4))
        
        angles = np.array([1.0, 0.8, -0.75])
        lines = VGroup()
        for i in range(3):
            pos = dots[i].get_center()
            s = np.sin(angles[i])
            c = np.cos(angles[i])

            start = pos - 0.5 * c * RIGHT - 0.5 * s * UP
            end = pos + 0.5 * c * RIGHT + 0.5 * s * UP

            l = DashedLine(start, end, color=color1, stroke_width = 0.6 * DEFAULT_STROKE_WIDTH)
            lines.add(l)
        
        beziers = VGroup()
        for i in range(2):
            sa = dots[i].get_center()
            sh = sa + 2 * lines[i].get_unit_vector()
            ea = dots[i+1].get_center()
            eh = ea - 2 * lines[i+1].get_unit_vector()
            b = CubicBezier(sa, sh, eh, ea, color=color2)
            beziers.add(b)
        
        return dots, lines, beziers
    
    def octaves(self, lac_val, pers_val, octaves=3):
        perlin_axes_main = Axes(x_range=[0, 1], y_range=[-1, 1], x_length=12, y_length=2, color=color2)
        perlin_axes_main.shift(1.5 * UP)

        perlin_axes_temp = Axes(x_range=[0, 1], y_range=[-1, 1], x_length=12, y_length=2, color=color2)
        perlin_axes_temp.shift(2.5 * DOWN)
        
        x_vals = np.linspace(0, 1, 1001)
        y_vals = np.zeros_like(x_vals)

        tot_noise = Text("Total Noise: ", color=color3, font_size=28).to_edge(LEFT).shift(3 * UP)
        curr_noise = Text("Current Octave: ", color=color3, font_size=28).to_edge(LEFT).shift(DOWN)

        cite = Tex("Perlin 1984; Perlin 1985; Perlin and Hoffert 1989", font_size=20, color=color4).to_corner(DR)

        self.play(Write(tot_noise), Write(curr_noise), Write(cite))

        perlin_plot_main = perlin_axes_main.plot_line_graph(x_values=x_vals, y_values=y_vals, line_color=color2, \
                                                            add_vertex_dots=False, stroke_width=0.6 * DEFAULT_STROKE_WIDTH)
        perlin_plot_temp = None

        self.play(Create(perlin_plot_main))
        self.next_slide()

        overlay = Rectangle(color=color5, height=9, width=16, stroke_width=0).set_opacity(0.9)
        self.play(FadeIn(overlay))
        lac = Text("Lacunarity: ", color=color1, font_size=28).shift(2 * UP + 4 * LEFT)
        lac_def = Tex("The ratio of successive frequencies of a fractal noise function.", font_size=28, color=color2).next_to(lac, RIGHT)
        
        pers = Text("Persistence: ", color=color1, font_size=28).shift(2 * DOWN + 4 * LEFT)
        pers_def = Tex("The ratio of successive amplitudes of a fractal noise function.", font_size=28, color=color2).next_to(pers, RIGHT)

        self.play(Write(lac), Write(pers))
        self.next_slide()
        self.play(Write(lac_def))
        self.next_slide()
        self.play(Write(pers_def))
        self.next_slide()

        self.play(Uncreate(lac_def), Uncreate(pers_def))
        self.play(lac.animate.move_to(6 * RIGHT + 0.5 * UP).scale(0.6), pers.animate.move_to(6 * RIGHT).scale(0.6))
        self.play(FadeOut(overlay))
        self.next_slide()

        lac_num = DecimalNumber(2, font_size=28, color=color2).next_to(lac)
        pers_num = DecimalNumber(0.5, font_size=28, color=color2).next_to(pers)
        self.play(Write(lac_num), Write(pers_num))
        self.next_slide()

        for i in range(octaves):
            noise = PerlinNoise(10 * (lac_val ** i), seed=32422)
            temp = np.array([noise(x) for x in x_vals])

            perlin_plot_temp = perlin_axes_temp.plot_line_graph(x_values=x_vals, y_values=temp, line_color=color2, \
                                                           add_vertex_dots=False, stroke_width=0.6 * DEFAULT_STROKE_WIDTH)
            self.play(Create(perlin_plot_temp))
            self.next_slide()

            y_vals += (pers_val ** i) * temp

            temp_plot = perlin_axes_main.plot_line_graph(x_values=x_vals, y_values=y_vals, line_color=color2, \
                                                        add_vertex_dots=False, stroke_width=0.6 * DEFAULT_STROKE_WIDTH)
            self.play(ReplacementTransform(VGroup(perlin_plot_temp, perlin_plot_main), temp_plot))
            perlin_plot_main = temp_plot
            self.next_slide()
        
        self.play(Uncreate(lac), Uncreate(pers), Uncreate(lac_num), Uncreate(pers_num), Uncreate(perlin_plot_main), Uncreate(tot_noise), Uncreate(curr_noise), Uncreate(cite))
        self.next_slide()
    
    def perlin_2d_demo(self):
        sq_main = Square(4, stroke_color=color1, fill_opacity=0)
        self.play(DrawBorderThenFill(sq_main))
        cite = Tex("Perlin 1984; Perlin 1985; Perlin and Hoffert 1989", font_size=20, color=color4).to_corner(DR)
        self.play(Write(cite))
        self.next_slide()

        sq_1 = Square(3, stroke_color=color1, fill_opacity=0).move_to(4 * LEFT + 2 * DOWN)
        sq_2 = Square(3, stroke_color=color1, fill_opacity=0).move_to(4 * RIGHT + 2 * DOWN)
        sq_3 = Square(3, stroke_color=color1, fill_opacity=0).move_to(4 * LEFT + 2 * UP)
        sq_4 = Square(3, stroke_color=color1, fill_opacity=0).move_to(4 * RIGHT + 2 * UP)

        squares = VGroup(sq_1, sq_2, sq_3, sq_4)

        self.play(ReplacementTransform(sq_main.copy(), sq_1), ReplacementTransform(sq_main.copy(), sq_2), ReplacementTransform(sq_main.copy(), sq_3), ReplacementTransform(sq_main, sq_4))
        self.next_slide()

        angles = rng.random(size=(4, )) * np.pi * 2 - np.pi
        vecs = VGroup()
        for i, alp in enumerate(angles):
            x = (i % 2) * 2 - 1
            y = (i // 2) * 2 - 1

            pos = squares[i].get_corner(x * RIGHT + y * UP)
            v = Vector(np.cos(alp) * RIGHT + np.sin(alp) * UP, color=color3, z_index=1).shift(pos)
            vecs.add(v)
        
        self.play(Create(vecs))
        self.next_slide()

        perlin_sq = VGroup()
        for i in range(4):
            curr_corx = (i % 2) * 2 - 1
            curr_cory = (i // 2) * 2 - 1
            
            vec_pos = squares[i].get_corner(curr_corx * RIGHT + curr_cory * UP)
            for y_idx in range(20):
                for x_idx in range(20):
                    x = (x_idx * 3 + 1.5) / 20
                    y = (y_idx * 3 + 1.5) / 20

                    dot_pos = squares[i].get_corner(- RIGHT - UP) + x * RIGHT + y * UP
                    dot_vec = dot_pos - vec_pos

                    dot_vec /= np.linalg.norm(dot_vec)

                    dot_col = dot_vec[0] * np.cos(angles[i]) + dot_vec[1] * np.sin(angles[i])
                    r = dot_col * 255 if dot_col > 0 else 0
                    g = - dot_col * 255 if dot_col < 0 else 0
                    b = 0

                    sq = Square(3.0/20, fill_color=ManimColor([r, g, b]), stroke_width=0, fill_opacity=1).move_to(dot_pos)
                    perlin_sq.add(sq)

        self.play(FadeIn(perlin_sq))
        self.next_slide()

        perlin_final_sq = VGroup()
        for i in range(400):
            sq1_col = perlin_sq[       i].get_color().to_rgb() / 255.0
            sq2_col = perlin_sq[ 400 + i].get_color().to_rgb() / 255.0
            sq3_col = perlin_sq[ 800 + i].get_color().to_rgb() / 255.0
            sq4_col = perlin_sq[1200 + i].get_color().to_rgb() / 255.0

            sq1_col = sq1_col[0] if sq1_col[0] != 0 else -sq1_col[1]
            sq2_col = sq2_col[0] if sq2_col[0] != 0 else -sq2_col[1]
            sq3_col = sq3_col[0] if sq3_col[0] != 0 else -sq3_col[1]
            sq4_col = sq4_col[0] if sq4_col[0] != 0 else -sq4_col[1]

            x = (i % 20) / 20.0
            y = (i // 20) / 20.0
            val1 = self.interp_cubic(sq1_col, sq2_col, x)
            val2 = self.interp_cubic(sq3_col, sq4_col, x)

            col = self.interp_cubic(val1, val2, y) * 255

            sq = Square(4 / 20, fill_color=ManimColor([col, col, col]), stroke_width=0, fill_opacity=1).move_to((4 * x - 2) * RIGHT + (4 * y - 2) * UP)
            perlin_final_sq.add(sq)
        
        self.play(FadeOut(squares), FadeOut(vecs), ReplacementTransform(perlin_sq, perlin_final_sq))
        self.next_slide()
        self.play(FadeOut(perlin_final_sq), Uncreate(cite))
    
    def interp_cubic(self, val1, val2, dx):
        return (val2 - val1) * (3.0 - 2.0 * dx) * dx * dx + val1
    
    def perlin_octaves(self, lac_val, pers_val, octaves=3):
        self.next_slide(loop=True)

        colors = np.zeros((100, 100))

        sq = VGroup()
        for y_idx in range(100):
            for x_idx in range(100):
                x = (x_idx + 0.5) / 100.0
                y = (y_idx + 0.5) / 100.0
                col = 0.0
                colors[y_idx, x_idx] = 0.0

                new_sq = Square(0.07, fill_color=ManimColor([0.5, 0.5, 0.5]), stroke_width=0, fill_opacity=1).move_to((x_idx * 0.07 - 3.5) * RIGHT + (y_idx * 0.07 - 3.5) * UP)
                sq.add(new_sq)
        
        self.play(FadeIn(sq))

        for i in range(octaves):
            temp = VGroup()
            noise = PerlinNoise(10 * (lac_val ** i), seed=32422)
            for y_idx in range(100):
                for x_idx in range(100):
                    x = (x_idx + 0.5) / 100.0
                    y = (y_idx + 0.5) / 100.0
                    col = noise([x, y])

                    colors[y_idx, x_idx] += (pers_val ** i) * col
                    col = colors[y_idx, x_idx] + 0.5

                    temp.add(sq[y_idx * 100 + x_idx].copy().set_color(ManimColor([col, col, col])))

            self.play(FadeOut(sq), FadeIn(temp))
            del sq
            sq = temp
            self.wait(1)
        
        self.next_slide()

        self.play(FadeOut(sq))
        self.next_slide()