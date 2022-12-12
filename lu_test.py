import numpy as np
from numpy.testing import assert_allclose
import unittest

import lu as submission


class LUTest(unittest.TestCase):
    def check_lu_common(
        self, A, L_ans, U_ans, P_ans=np.array([None]), Q_ans=np.array([None])
    ):
        if not P_ans.any():
            P_ans = np.eye(A.shape[0])
        if not Q_ans.any():
            Q_ans = np.eye(A.shape[0])

        self.assertTupleEqual(
            P_ans.shape, A.shape, msg="Err: P has a different shape than A"
        )
        self.assertTupleEqual(
            Q_ans.shape, A.shape, msg="Err: Q has a different shape than A"
        )
        self.assertTupleEqual(
            L_ans.shape, A.shape, msg="Err: L has a different shape than A"
        )
        self.assertTupleEqual(
            U_ans.shape, A.shape, msg="Err: U has a different shape than A"
        )
        assert_allclose(
            P_ans @ P_ans.T.conjugate(),
            np.eye(P_ans.shape[0]),
            atol=1e-14,
            rtol=1e-14,
            err_msg="Err: P is not orthogonal",
            verbose=False,
        )
        assert_allclose(
            Q_ans @ Q_ans.T.conjugate(),
            np.eye(Q_ans.shape[0]),
            atol=1e-14,
            rtol=1e-14,
            err_msg="Err: Q is not orthogonal",
            verbose=False,
        )
        assert_allclose(
            L_ans,
            np.tril(L_ans),
            atol=1e-14,
            rtol=1e-14,
            err_msg="Err: L is not lower triangular",
            verbose=False,
        )
        assert_allclose(
            U_ans,
            np.triu(U_ans),
            atol=1e-14,
            rtol=1e-14,
            err_msg="Err: U is not upper triangular",
            verbose=False,
        )

    def check_lu(self, A):
        with self.subTest(m=A.shape[0], dtype=A.dtype):
            L_ans, U_ans = submission.lu(A.copy())
            self.check_lu_common(A, L_ans, U_ans)

            assert_allclose(
                A,
                L_ans @ U_ans,
                atol=1e-10,
                rtol=1e-10,
                err_msg="Err: A != LU",
                verbose=False,
            )

    def check_lu_complete(self, A):
        with self.subTest(m=A.shape[0], dtype=A.dtype):
            P_ans, Q_ans, L_ans, U_ans = submission.lu_complete(A.copy())
            self.check_lu_common(A, L_ans, U_ans, P_ans, Q_ans)
            assert_allclose(
                P_ans @ A @ Q_ans,
                L_ans @ U_ans,
                atol=1e-10,
                rtol=1e-10,
                err_msg="Err: PAQ != LU",
                verbose=False,
            )

    def check_lu_pivoting_scheme(self, A):
        with self.subTest(m=A.shape[0], dtype=A.dtype):
            i, j = submission.maxabs_idx(A.copy())
            assert_allclose(
                np.abs(A[i, j]),
                (np.abs(A)).max(),
                atol=1e-13,
                rtol=1e-13,
                err_msg="Err: Wrong pivot elemet chosen",
                verbose=False,
            )

    def test_01(self):
        A = np.eye(4)
        self.check_lu(A)
        self.check_lu_complete(A)
        self.check_lu_pivoting_scheme(A)

    # todo: implement more tests
    # def test_02(self):
    #     A = ...
    #     self.check_lu(A)
    #     self.check_lu_complete(A)
    #     self.check_lu_pivoting_scheme(A)
    #
    # def test_03(self)...

if __name__ == "__main__":
    np.random.seed(4)
    unittest.main(verbosity=2)
