from itertools import product, chain

from .klrw_algebra import KLRWAlgebra


class KLRWAlgebraWithTests(KLRWAlgebra):
    def check(self):
        print("Checking linearity of the right dot action.")
        self.check_linearity_of_right_dots_action_over_center()
        print("Checking relations.")
        self.check_relations()
        print("Checking one.")
        self.check_one()
        print("Checking well-definedness of multiplication.")
        self.check_welldef(self.basis_over_center(), self.basis_over_center())
        print("Checking action on dots.")
        self.check_associativity(
            self.basis_over_dots_and_center(),
            self.gens_over_dots(),
            self.basis_in_dots_modulo_center(),
        )

    def check_one(self):
        one = self._one_()
        count = 0
        for x in self.basis_over_center():
            assert one * x == x
            assert x * one == x
            count += 1
            if count % 1000000 == 0:
                print("So far checked ", count, " relations.")
        print("Elements checked:", count)

    def check_welldef(
        self,
        first_family,
        second_family,
        check_right_state=True,
        check_left_state=True,
        check_braids=False,
        check_degrees=True,
    ):
        """
        Checks that multiplication is well-defined
        """
        count = 0
        for a, b in product(first_family, second_family):
            ab = a * b
            count += 1
            if check_right_state:
                for braid_ab, __ in ab:
                    for braid_b, _ in b:
                        assert (
                            braid_ab.right_state() == braid_b.right_state()
                        ), "Right state don't match for {} braid in {} = {}*{}".format(
                            braid_ab, ab, a, b
                        )
            if check_left_state:
                for braid_ab, __ in ab:
                    for braid_a, _ in a:
                        assert (
                            braid_ab.left_state() == braid_a.left_state()
                        ), "Left state don't match for {} in {} = {}*{}".format(
                            braid_ab, ab, a, b
                        )
            if check_braids:
                raise NotImplementedError()
            if check_degrees:
                for braid_ab, coeff_ab in ab:
                    for braid_a, coeff_a in a:
                        for braid_b, coeff_b in b:
                            term_ab = self.term(braid_ab, coeff_ab)
                            term_a = self.term(braid_a, coeff_a)
                            term_b = self.term(braid_b, coeff_b)
                            degree_ab = term_ab.degree(check_if_homogeneous=True)
                            degree_a = term_a.degree(check_if_homogeneous=True)
                            degree_b = term_b.degree(check_if_homogeneous=True)
                            assert (
                                degree_ab == degree_a + degree_b
                            ), "Degrees don't match for {} in {} = {}*{}".format(
                                term_ab, ab, a, b
                            )
            if count % 1000000 == 0:
                print("So far checked ", count, " relations.")

        print("Products checked:", count)

    def check_linearity_of_right_dots_action_over_center(self):
        """
        Checks whether
        braid*(center*dots) = (center*braid)*dots
        In particular, for dots = 1 this checks commutativity with the center.
        The check assumes that the action is additive.
        """
        count = 0
        iterator_over_elements_in_center = chain(
            [self.base().one(), -self.base().one(), 2 * self.base().one()],
            self.center_gens(),
        )
        # print(sum(1 for _ in iterator_over_elements_in_center))
        # print(sum(1 for _ in self.basis_over_center()))
        # print(sum(1 for _ in self.basis_in_dots_modulo_center()))
        for x, c, d in product(
            self.basis_over_center(),
            iterator_over_elements_in_center,
            self.basis_in_dots_modulo_center(),
        ):
            assert (
                x * (c * d) == (c * x) * d
            ), "Issue with {0}*({1}*{2}) == ({1}*{0})*{2}".format(x, c, d)
            count += 1
            if count % 1000000 == 0:
                print("So far checked ", count, " relations.")

        print("Relations checked:", count)

    def check_relations(self):
        """
        Checks that the defining relations hold.
        Relation xx means relation in formula (2.xx)
        in https://arxiv.org/pdf/1111.1431.pdf
        Letters are added if there are several pictures in one formula
        """
        states = self.KLRWBraid.KLRWstate_set

        cases8a = 0
        for state in states:
            for i in range(len(state) - 1):
                if state[i] == state[i + 1]:
                    if not state[i].is_framing():
                        x = self.KLRWmonomial(state=state, word=(i + 1,))
                        assert (x * x).is_zero()
                        cases8a += 1
        print("Relation 8a holds! Cases checked:", cases8a)

        cases8b = 0
        for state in self.KLRWBraid.KLRWstate_set:
            for i in range(len(state) - 2):
                if state[i] == state[i + 1] and state[i + 1] == state[i + 2]:
                    if not state[i].is_framing():
                        x = self.KLRWmonomial(state=state, word=(i + 1,))
                        y = self.KLRWmonomial(state=state, word=(i + 2,))
                        assert x * y * x == y * x * y, print(
                            "Relation 8b does not hold: "
                            + "{0}*{1}*{0} != {1}*{0}*{1}".format(x, y)
                        )
                        assert x * y * x != self.zero()
                        cases8b += 1
        print("Relation 8b holds! Cases checked:", cases8b)

        cases8c = 0
        for state in states:
            for i in range(len(state) - 1):
                if state[i] == state[i + 1]:
                    if not state[i].is_framing():
                        dot_index = state.index_among_same_color(i)
                        dot_left = (
                            self.base().dot_variable(state[i], dot_index).monomial
                        )
                        dot_right = (
                            self.base().dot_variable(state[i], dot_index + 1).monomial
                        )
                        r = self.base().vertex_variable(state[i]).monomial
                        x = self.KLRWmonomial(state=state, word=(i + 1,))
                        # print(r.__class__, dot_left.__class__, dot_right.__class__)
                        # print(dot_left*x - x*dot_right, r*self.idempotent(state))
                        # for braid,coeff in dot_left*x:
                        #    print(braid, coeff, coeff.__class__)
                        assert dot_left * x - x * dot_right == r * self.idempotent(
                            state
                        ), "{0}*{1}-{1}*{2} != {3}".format(
                            dot_left, x, dot_right, r * self.idempotent(state)
                        )
                        assert dot_left * x != self.zero() or dot_left.is_zero()
                        assert x * dot_right != self.zero() or dot_right.is_zero()
                        cases8c += 1
        print("Relation 8c holds! Cases checked:", cases8c)

        cases8d = 0
        for state in states:
            for i in range(len(state) - 1):
                if state[i] == state[i + 1]:
                    if not state[i].is_framing():
                        dot_index = state.index_among_same_color(i)
                        dot_left = (
                            self.base().dot_variable(state[i], dot_index).monomial
                        )
                        dot_right = (
                            self.base().dot_variable(state[i], dot_index + 1).monomial
                        )
                        r = self.base().vertex_variable(state[i]).monomial
                        x = self.KLRWmonomial(state=state, word=(i + 1,))
                        assert x * dot_left - dot_right * x == r * self.idempotent(
                            state
                        )
                        assert x * dot_left != self.zero() or dot_left.is_zero()
                        assert dot_right * x != self.zero() or dot_right.is_zero()
                        cases8d += 1
        print("Relation 8d holds! Cases checked:", cases8d)

        cases10 = 0
        for state in states:
            for i in range(len(state) - 1):
                if state[i] != state[i + 1]:
                    if not state[i].is_framing() or not state[i + 1].is_framing():
                        d_ij = -self.quiver[state[i], state[i + 1]]
                        t_ij = (
                            self.base().edge_variable(state[i], state[i + 1]).monomial
                        )
                        if d_ij > 0:
                            d_ji = -self.quiver[state[i + 1], state[i]]
                            t_ji = (
                                self.base()
                                .edge_variable(state[i + 1], state[i])
                                .monomial
                            )
                            dot_left_index = state.index_among_same_color(i)
                            dot_right_index = state.index_among_same_color(i + 1)
                            dot_left = (
                                self.base()
                                .dot_variable(state[i], dot_left_index)
                                .monomial
                            )
                            dot_right = (
                                self.base()
                                .dot_variable(state[i + 1], dot_right_index)
                                .monomial
                            )
                            coeff = t_ij * (dot_left**d_ij) + t_ji * (dot_right**d_ji)
                        else:
                            coeff = t_ij
                        x1 = self.KLRWmonomial(
                            state=state.act_by_s(i + 1), word=(i + 1,)
                        )
                        x2 = self.KLRWmonomial(state=state, word=(i + 1,))
                        assert x1 * x2 == coeff * self.idempotent(state), print(
                            "Relation 9 does not hold: {0}*{1} == {2} != {3}".format(
                                x1, x2, x1 * x2, coeff * self.idempotent(state)
                            )
                        )
                        assert x1 * x2 != self.zero()
                        cases10 += 1
        print("Relation 10 holds! Cases checked:", cases10)

        cases12a = 0
        for state in states:
            for i in range(len(state) - 1):
                if state[i] != state[i + 1]:
                    if not state[i].is_framing() or not state[i + 1].is_framing():
                        dot_index = state.index_among_same_color(i + 1)
                        dot = self.base().dot_variable(state[i + 1], dot_index).monomial
                        x = self.KLRWmonomial(state=state, word=(i + 1,))
                        assert dot * x == x * dot
                        assert dot * x != self.zero() or dot.is_zero()
                        cases12a += 1
        print("Relation 12a holds! Cases checked:", cases12a)

        cases12b = 0
        for state in states:
            for i in range(len(state) - 1):
                if state[i] != state[i + 1]:
                    if not state[i].is_framing() or not state[i + 1].is_framing():
                        dot_index = state.index_among_same_color(i)
                        dot = self.base().dot_variable(state[i], dot_index).monomial
                        x = self.KLRWmonomial(state=state, word=(i + 1,))
                        assert dot * x == x * dot
                        assert dot * x != self.zero() or dot.is_zero()
                        cases12b += 1
        print("Relation 12b holds! Cases checked:", cases12b)

        cases13 = 0
        for state in self.KLRWBraid.KLRWstate_set:
            for i in range(len(state) - 2):
                if (
                    state[i] != state[i + 2]
                    or -self.quiver[state[i], state[i + 1]] <= 0
                ):
                    number_of_framings = sum(
                        1 for color in state[i : i + 3] if color.is_framing()
                    )
                    if number_of_framings <= 1:
                        # print(state, i)
                        x3 = self.KLRWmonomial(state=state, word=(i + 1,))
                        x2 = self.KLRWmonomial(
                            state=state.act_by_s(i + 1), word=(i + 2,)
                        )
                        x1 = self.KLRWmonomial(
                            state=state.act_by_s(i + 1).act_by_s(i + 2), word=(i + 1,)
                        )
                        y3 = self.KLRWmonomial(state=state, word=(i + 2,))
                        y2 = self.KLRWmonomial(
                            state=state.act_by_s(i + 2), word=(i + 1,)
                        )
                        y1 = self.KLRWmonomial(
                            state=state.act_by_s(i + 2).act_by_s(i + 1), word=(i + 2,)
                        )
                        assert x1 * x2 * x3 == y1 * y2 * y3, print(
                            "Relation 13 does not hold: "
                            + "{0}*{1}*{2} = {3} != {4} = {5}*{6}*{7}".format(
                                x1, x2, x3, x1 * x2 * x3, y1 * y2 * y3, y1, y2, y3
                            )
                        )
                        assert x1 * x2 * x3 != self.zero()
                        cases13 += 1
        print("Relation 13 holds! Cases checked:", cases13)

        cases14 = 0
        for state in self.KLRWBraid.KLRWstate_set:
            for i in range(len(state) - 2):
                if (
                    state[i] == state[i + 2]
                    and -self.quiver[state[i], state[i + 1]] > 0
                ):
                    if not state[i].is_framing():
                        x3 = self.KLRWmonomial(state=state, word=(i + 1,))
                        x2 = self.KLRWmonomial(
                            state=state.act_by_s(i + 1), word=(i + 2,)
                        )
                        x1 = self.KLRWmonomial(
                            state=state.act_by_s(i + 1).act_by_s(i + 2), word=(i + 1,)
                        )
                        y3 = self.KLRWmonomial(state=state, word=(i + 2,))
                        y2 = self.KLRWmonomial(
                            state=state.act_by_s(i + 2), word=(i + 1,)
                        )
                        y1 = self.KLRWmonomial(
                            state=state.act_by_s(i + 2).act_by_s(i + 1), word=(i + 2,)
                        )
                        r_i = self.base().vertex_variable(state[i]).monomial
                        d_ij = -self.quiver[state[i], state[i + 1]]
                        t_ij = (
                            self.base().edge_variable(state[i], state[i + 1]).monomial
                        )
                        dot_left_index = state.index_among_same_color(i)
                        dot_left = (
                            self.base().dot_variable(state[i], dot_left_index).monomial
                        )
                        dot_right = (
                            self.base()
                            .dot_variable(state[i], dot_left_index + 1)
                            .monomial
                        )
                        coeff = (
                            t_ij
                            * r_i
                            * sum(
                                dot_left**k * dot_right ** (d_ij - k - 1)
                                for k in range(d_ij)
                            )
                        )
                        assert x1 * x2 * x3 - y1 * y2 * y3 == coeff * self.idempotent(
                            state
                        ), print(
                            "Relation 14 does not hold: {0} - {1} != {2}".format(
                                x1 * x2 * x3,
                                y1 * y2 * y3,
                                coeff * self.idempotent(state),
                            )
                        )
                        assert x1 * x2 * x3 != self.zero()
                        cases14 += 1
        print("Relation 14 holds! Cases checked:", cases14)

    def check_associativity(self, first_family, second_family, third_family):
        """
        Checks associativity
        If a family is not given, it will check for all basis vectors
        If max_word_length is given, for each family only elements
        that have all terms of length less than this number are considered
        """

        count = 0
        for a, b, c in product(first_family, second_family, third_family):
            assert a * (b * c) == (a * b) * c, (
                "Issue with {}, {}, {}\n".format(a, b, c)
                + "bc = {}\n".format(b * c)
                + "ab= {}\n".format(a * b)
            )
            count += 1
            if count % 1000000 == 0:
                print("So far checked ", count, " relations.")

        print("Relations checked:", count)
