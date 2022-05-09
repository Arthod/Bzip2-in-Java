
return
import warnings
import itertools
import sys

import pytest

import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_array_almost_equal, assert_no_warnings,
    assert_allclose, HAS_REFCOUNT, suppress_warnings
    )
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle


UNARY_UFUNCS = [obj for obj in np.core.umath.__dict__.values()
                    if isinstance(obj, np.ufunc)]
UNARY_OBJECT_UFUNCS = [uf for uf in UNARY_UFUNCS if "O->O" in uf.types]


class TestUfuncKwargs:
    def test_kwarg_exact(self):
        assert_raises(TypeError, np.add, 1, 2, castingx='safe')
        assert_raises(TypeError, np.add, 1, 2, dtypex=int)
        assert_raises(TypeError, np.add, 1, 2, extobjx=[4096])
        assert_raises(TypeError, np.add, 1, 2, outx=None)
        assert_raises(TypeError, np.add, 1, 2, sigx='ii->i')
        assert_raises(TypeError, np.add, 1, 2, signaturex='ii->i')
        assert_raises(TypeError, np.add, 1, 2, subokx=False)
        assert_raises(TypeError, np.add, 1, 2, wherex=[True])

    def test_sig_signature(self):
        assert_raises(TypeError, np.add, 1, 2, sig='ii->i',
                      signature='ii->i')

    def test_sig_dtype(self):
        assert_raises(TypeError, np.add, 1, 2, sig='ii->i',
                      dtype=int)
        assert_raises(TypeError, np.add, 1, 2, signature='ii->i',
                      dtype=int)

    def test_extobj_refcount(self):
        # Should not segfault with USE_DEBUG.
        assert_raises(TypeError, np.add, 1, 2, extobj=[4096], parrot=True)


class TestUfuncGenericLoops:
    """Test generic loops.

    The loops to be tested are:

        PyUFunc_ff_f_As_dd_d
        PyUFunc_ff_f
        PyUFunc_dd_d
        PyUFunc_gg_g
        PyUFunc_FF_F_As_DD_D
        PyUFunc_DD_D
        PyUFunc_FF_F
        PyUFunc_GG_G
        PyUFunc_OO_O
        PyUFunc_OO_O_method
        PyUFunc_f_f_As_d_d
        PyUFunc_d_d
        PyUFunc_f_f
        PyUFunc_g_g
        PyUFunc_F_F_As_D_D
        PyUFunc_F_F
        PyUFunc_D_D
        PyUFunc_G_G
        PyUFunc_O_O
        PyUFunc_O_O_method
        PyUFunc_On_Om

    Where:

        f -- float
        d -- double
        g -- long double
        F -- complex float
        D -- complex double
        G -- complex long double
        O -- python object

    It is difficult to assure that each of these loops is entered from the
    Python level as the special cased loops are a moving target and the
    corresponding types are architecture dependent. We probably need to
    define C level testing ufuncs to get at them. For the time being, I've
    just looked at the signatures registered in the build directory to find
    relevant functions.

    """
    np_dtypes = [
        (np.single, np.single), (np.single, np.double),
        (np.csingle, np.csingle), (np.csingle, np.cdouble),
        (np.double, np.double), (np.longdouble, np.longdouble),
        (np.cdouble, np.cdouble), (np.clongdouble, np.clongdouble)]

    @pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
    def test_unary_PyUFunc(self, input_dtype, output_dtype, f=np.exp, x=0, y=1):
        xs = np.full(10, input_dtype(x), dtype=output_dtype)
        ys = f(xs)[::2]
        assert_allclose(ys, y)
        assert_equal(ys.dtype, output_dtype)

    def f2(x, y):
        return x**y

    @pytest.mark.parametrize('input_dtype,output_dtype', np_dtypes)
    def test_binary_PyUFunc(self, input_dtype, output_dtype, f=f2, x=0, y=1):
        xs = np.full(10, input_dtype(x), dtype=output_dtype)
        ys = f(xs, xs)[::2]
        assert_allclose(ys, y)
        assert_equal(ys.dtype, output_dtype)

    # class to use in testing object method loops
    class foo:
        def conjugate(self):
            return np.bool_(1)

        def logical_xor(self, obj):
            return np.bool_(1)

    def test_unary_PyUFunc_O_O(self):
        x = np.ones(10, dtype=object)
        assert_(np.all(np.abs(x) == 1))

    def test_unary_PyUFunc_O_O_method_simple(self, foo=foo):
        x = np.full(10, foo(), dtype=object)
        assert_(np.all(np.conjugate(x) == True))

    def test_binary_PyUFunc_OO_O(self):
        x = np.ones(10, dtype=object)
        assert_(np.all(np.add(x, x) == 2))

    def test_binary_PyUFunc_OO_O_method(self, foo=foo):
        x = np.full(10, foo(), dtype=object)
        assert_(np.all(np.logical_xor(x, x)))

    def test_binary_PyUFunc_On_Om_method(self, foo=foo):
        x = np.full((10, 2, 3), foo(), dtype=object)
        assert_(np.all(np.logical_xor(x, x)))

    def test_python_complex_conjugate(self):
        # The conjugate ufunc should fall back to calling the method:
        arr = np.array([1+2j, 3-4j], dtype="O")
        assert isinstance(arr[0], complex)
        res = np.conjugate(arr)
        assert res.dtype == np.dtype("O")
        assert_array_equal(res, np.array([1-2j, 3+4j], dtype="O"))

    @pytest.mark.parametrize("ufunc", UNARY_OBJECT_UFUNCS)
    def test_unary_PyUFunc_O_O_method_full(self, ufunc):
        """Compare the result of the object loop with non-object one"""
        val = np.float64(np.pi/4)

        class MyFloat(np.float64):
            def __getattr__(self, attr):
                try:
                    return super().__getattr__(attr)
                except AttributeError:
                    return lambda: getattr(np.core.umath, attr)(val)

        # Use 0-D arrays, to ensure the same element call
        num_arr = np.array(val, dtype=np.float64)
        obj_arr = np.array(MyFloat(val), dtype="O")

        with np.errstate(all="raise"):
            try:
                res_num = ufunc(num_arr)
            except Exception as exc:
                with assert_raises(type(exc)):
                    ufunc(obj_arr)
            else:
                res_obj = ufunc(obj_arr)
                assert_array_almost_equal(res_num.astype("O"), res_obj)


def _pickleable_module_global():
    pass


class TestUfunc:
    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_(pickle.loads(pickle.dumps(np.sin,
                                              protocol=proto)) is np.sin)

            # Check that ufunc not defined in the top level numpy namespace
            # such as numpy.core._rational_tests.test_add can also be pickled
            res = pickle.loads(pickle.dumps(_rational_tests.test_add,
                                            protocol=proto))
            assert_(res is _rational_tests.test_add)

    def test_pickle_withstring(self):
        astring = (b"cnumpy.core\n_ufunc_reconstruct\np0\n"
                   b"(S'numpy.core.umath'\np1\nS'cos'\np2\ntp3\nRp4\n.")
        assert_(pickle.loads(astring) is np.cos)

    def test_pickle_name_is_qualname(self):
        # This tests that a simplification of our ufunc pickle code will
        # lead to allowing qualnames as names.  Future ufuncs should
        # possible add a specific qualname, or a hook into pickling instead
        # (dask+numba may benefit).
        _pickleable_module_global.ufunc = umt._pickleable_module_global_ufunc
        obj = pickle.loads(pickle.dumps(_pickleable_module_global.ufunc))
        assert obj is umt._pickleable_module_global_ufunc

    def test_reduceat_shifting_sum(self):
        L = 6
        x = np.arange(L)
        idx = np.array(list(zip(np.arange(L - 2), np.arange(L - 2) + 2))).ravel()
        assert_array_equal(np.add.reduceat(x, idx)[::2], [1, 3, 5, 7])

    def test_all_ufunc(self):
        """Try to check presence and results of all ufuncs.

        The list of ufuncs comes from generate_umath.py and is as follows:

        =====  ====  =============  ===============  ========================
        done   args   function        types                notes
        =====  ====  =============  ===============  ========================
        n      1     conjugate      nums + O
        n      1     absolute       nums + O         complex -> real
        n      1     negative       nums + O
        n      1     sign           nums + O         -> int
        n      1     invert         bool + ints + O  flts raise an error
        n      1     degrees        real + M         cmplx raise an error
        n      1     radians        real + M         cmplx raise an error
        n      1     arccos         flts + M
        n      1     arccosh        flts + M
        n      1     arcsin         flts + M
        n      1     arcsinh        flts + M
        n      1     arctan         flts + M
        n      1     arctanh        flts + M
        n      1     cos            flts + M
        n      1     sin            flts + M
        n      1     tan            flts + M
        n      1     cosh           flts + M
        n      1     sinh           flts + M
        n      1     tanh           flts + M
        n      1     exp            flts + M
        n      1     expm1          flts + M
        n      1     log            flts + M
        n      1     log10          flts + M
        n      1     log1p          flts + M
        n      1     sqrt           flts + M         real x < 0 raises error
        n      1     ceil           real + M
        n      1     trunc          real + M
        n      1     floor          real + M
        n      1     fabs           real + M
        n      1     rint           flts + M
        n      1     isnan          flts             -> bool
        n      1     isinf          flts             -> bool
        n      1     isfinite       flts             -> bool
        n      1     signbit        real             -> bool
        n      1     modf           real             -> (frac, int)
        n      1     logical_not    bool + nums + M  -> bool
        n      2     left_shift     ints + O         flts raise an error
        n      2     right_shift    ints + O         flts raise an error
        n      2     add            bool + nums + O  boolean + is ||
        n      2     subtract       bool + nums + O  boolean - is ^
        n      2     multiply       bool + nums + O  boolean * is &
        n      2     divide         nums + O
        n      2     floor_divide   nums + O
        n      2     true_divide    nums + O         bBhH -> f, iIlLqQ -> d
        n      2     fmod           nums + M
        n      2     power          nums + O
        n      2     greater        bool + nums + O  -> bool
        n      2     greater_equal  bool + nums + O  -> bool
        n      2     less           bool + nums + O  -> bool
        n      2     less_equal     bool + nums + O  -> bool
        n      2     equal          bool + nums + O  -> bool
        n      2     not_equal      bool + nums + O  -> bool
        n      2     logical_and    bool + nums + M  -> bool
        n      2     logical_or     bool + nums + M  -> bool
        n      2     logical_xor    bool + nums + M  -> bool
        n      2     maximum        bool + nums + O
        n      2     minimum        bool + nums + O
        n      2     bitwise_and    bool + ints + O  flts raise an error
        n      2     bitwise_or     bool + ints + O  flts raise an error
        n      2     bitwise_xor    bool + ints + O  flts raise an error
        n      2     arctan2        real + M
        n      2     remainder      ints + real + O
        n      2     hypot          real + M
        =====  ====  =============  ===============  ========================

        Types other than those listed will be accepted, but they are cast to
        the smallest compatible type for which the function is defined. The
        casting rules are:

        bool -> int8 -> float32
        ints -> double

        """
        pass

    # from include/numpy/ufuncobject.h
    size_inferred = 2
    can_ignore = 4
    def test_signature0(self):
        # the arguments to test_signature are: nin, nout, core_signature
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i),(i)->()")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1,  1,  0))
        assert_equal(ixs, (0, 0))
        assert_equal(flags, (self.size_inferred,))
        assert_equal(sizes, (-1,))

    def test_signature1(self):
        # empty core signature; treat as plain ufunc (with trivial core)
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(),()->()")
        assert_equal(enabled, 0)
        assert_equal(num_dims, (0,  0,  0))
        assert_equal(ixs, ())
        assert_equal(flags, ())
        assert_equal(sizes, ())

    def test_signature2(self):
        # more complicated names for variables
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(i1,i2),(J_1)->(_kAB)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 1, 1))
        assert_equal(ixs, (0, 1, 2, 3))
        assert_equal(flags, (self.size_inferred,)*4)
        assert_equal(sizes, (-1, -1, -1, -1))

    def test_signature3(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, u"(i1, i12),   (J_1)->(i12, i2)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 1, 2))
        assert_equal(ixs, (0, 1, 2, 1, 3))
        assert_equal(flags, (self.size_inferred,)*4)
        assert_equal(sizes, (-1, -1, -1, -1))

    def test_signature4(self):
        # matrix_multiply signature from _umath_tests
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(n,k),(k,m)->(n,m)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 2, 2))
        assert_equal(ixs, (0, 1, 1, 2, 0, 2))
        assert_equal(flags, (self.size_inferred,)*3)
        assert_equal(sizes, (-1, -1, -1))

    def test_signature5(self):
        # matmul signature from _umath_tests
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            2, 1, "(n?,k),(k,m?)->(n?,m?)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (2, 2, 2))
        assert_equal(ixs, (0, 1, 1, 2, 0, 2))
        assert_equal(flags, (self.size_inferred | self.can_ignore,
                             self.size_inferred,
                             self.size_inferred | self.can_ignore))
        assert_equal(sizes, (-1, -1, -1))

    def test_signature6(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            1, 1, "(3)->()")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 0))
        assert_equal(ixs, (0,))
        assert_equal(flags, (0,))
        assert_equal(sizes, (3,))

    def test_signature7(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "(3),(03,3),(n)->(9)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (0, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature8(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "(3?),(3?,3?),(n)->(9)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (self.can_ignore, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))
    
    def test_signature9(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            1, 1, "(  3)  -> ( )")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 0))
        assert_equal(ixs, (0,))
        assert_equal(flags, (0,))
        assert_equal(sizes, (3,))

    def test_signature10(self):
        enabled, num_dims, ixs, flags, sizes = umt.test_signature(
            3, 1, "( 3? ) , (3? ,  3?) ,(n )-> ( 9)")
        assert_equal(enabled, 1)
        assert_equal(num_dims, (1, 2, 1, 1))
        assert_equal(ixs, (0, 0, 0, 1, 2))
        assert_equal(flags, (self.can_ignore, self.size_inferred, 0))
        assert_equal(sizes, (3, -1, 9))

    def test_signature_failure_extra_parenthesis(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "((i)),(i)->()")

    def test_signature_failure_mismatching_parenthesis(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "(i),)i(->()")

    def test_signature_failure_signature_missing_input_arg(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 1, "(i),->()")

    def test_signature_failure_signature_missing_output_arg(self):
        with assert_raises(ValueError):
            umt.test_signature(2, 2, "(i),(i)->()")

    def test_get_signature(self):
        assert_equal(umt.inner1d.signature, "(i),(i)->()")

    def test_forced_sig(self):
        a = 0.5*np.arange(3, dtype='f8')
        assert_equal(np.add(a, 0.5), [0.5, 1, 1.5])
        with pytest.warns(DeprecationWarning):
            assert_equal(np.add(a, 0.5, sig='i', casting='unsafe'), [0, 0, 1])
        assert_equal(np.add(a, 0.5, sig='ii->i', casting='unsafe'), [0, 0, 1])
        with pytest.warns(DeprecationWarning):
            assert_equal(np.add(a, 0.5, sig=('i4',), casting='unsafe'),
                         [0, 0, 1])
        assert_equal(np.add(a, 0.5, sig=('i4', 'i4', 'i4'),
                                            casting='unsafe'), [0, 0, 1])

        b = np.zeros((3,), dtype='f8')
        np.add(a, 0.5, out=b)
        assert_equal(b, [0.5, 1, 1.5])
        b[:] = 0
        with pytest.warns(DeprecationWarning):
            np.add(a, 0.5, sig='i', out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a, 0.5, sig='ii->i', out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        with pytest.warns(DeprecationWarning):
            np.add(a, 0.5, sig=('i4',), out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])
        b[:] = 0
        np.add(a, 0.5, sig=('i4', 'i4', 'i4'), out=b, casting='unsafe')
        assert_equal(b, [0, 0, 1])

    def test_signature_all_None(self):
        # signature all None, is an acceptable alternative (since 1.21)
        # to not providing a signature.
        res1 = np.add([3], [4], sig=(None, None, None))
        res2 = np.add([3], [4])
        assert_array_equal(res1, res2)
        res1 = np.maximum([3], [4], sig=(None, None, None))
        res2 = np.maximum([3], [4])
        assert_array_equal(res1, res2)

        with pytest.raises(TypeError):
            # special case, that would be deprecated anyway, so errors:
            np.add(3, 4, signature=(None,))

    def test_signature_dtype_type(self):
        # Since that will be the normal behaviour (past NumPy 1.21)
        # we do support the types already:
        float_dtype = type(np.dtype(np.float64))
        np.add(3, 4, signature=(float_dtype, float_dtype, None))

    @pytest.mark.parametrize("casting", ["unsafe", "same_kind", "safe"])
    def test_partial_signature_mismatch(self, casting):
        # If the second argument matches already, no need to specify it:
        res = np.ldexp(np.float32(1.), np.int_(2), dtype="d")
        assert res.dtype == "d"
        res = np.ldexp(np.float32(1.), np.int_(2), signature=(None, None, "d"))
        assert res.dtype == "d"

        # ldexp only has a loop for long input as second argument, overriding
        # the output cannot help with that (no matter the casting)
        with pytest.raises(TypeError):
            np.ldexp(1., np.uint64(3), dtype="d")
        with pytest.raises(TypeError):
            np.ldexp(1., np.uint64(3), signature=(None, None, "d"))

    def test_use_output_signature_for_all_arguments(self):
        # Test that providing only `dtype=` or `signature=(None, None, dtype)`
        # is sufficient if falling back to a homogeneous signature works.
        # In this case, the `intp, intp -> intp` loop is chosen.
        res = np.power(1.5, 2.8, dtype=np.intp, casting="unsafe")
        assert res == 1  # the cast happens first.
        res = np.power(1.5, 2.8, signature=(None, None, np.intp),
                       casting="unsafe")
        assert res == 1
        with pytest.raises(TypeError):
            # the unsafe casting would normally cause errors though:
            np.power(1.5, 2.8, dtype=np.intp)

    def test_signature_errors(self):
        with pytest.raises(TypeError,
                    match="the signature object to ufunc must be a string or"):
            np.add(3, 4, signature=123.)  # neither a string nor a tuple

        with pytest.raises(ValueError):
            # bad symbols that do not translate to dtypes
            np.add(3, 4, signature="%^->#")

        with pytest.raises(ValueError):
            np.add(3, 4, signature=b"ii-i")  # incomplete and byte string

        with pytest.raises(ValueError):
            np.add(3, 4, signature="ii>i")  # incomplete string

        with pytest.raises(ValueError):
            np.add(3, 4, signature=(None, "f8"))  # bad length

        with pytest.raises(UnicodeDecodeError):
            np.add(3, 4, signature=b"\xff\xff->i")

    def test_forced_dtype_times(self):
        # Signatures only set the type numbers (not the actual loop dtypes)
        # so using `M` in a signature/dtype should generally work:
        a = np.array(['2010-01-02', '1999-03-14', '1833-03'], dtype='>M8[D]')
        np.maximum(a, a, dtype="M")
        np.maximum.reduce(a, dtype="M")

        arr = np.arange(10, dtype="m8[s]")
        np.add(arr, arr, dtype="m")
        np.maximum(arr, arr, dtype="m")

    @pytest.mark.parametrize("ufunc", [np.add, np.sqrt])
    def test_cast_safety(self, ufunc):
        """Basic test for the safest casts, because ufuncs inner loops can
        indicate a cast-safety as well (which is normally always "no").
        """
        def call_ufunc(arr, **kwargs):
            return ufunc(*(arr,) * ufunc.nin, **kwargs)

        arr = np.array([1., 2., 3.], dtype=np.float32)
        arr_bs = arr.astype(arr.dtype.newbyteorder())
        expected = call_ufunc(arr)
        # Normally, a "no" cast:
        res = call_ufunc(arr, casting="no")
        assert_array_equal(expected, res)
        # Byte-swapping is not allowed with "no" though:
        with pytest.raises(TypeError):
            call_ufunc(arr_bs, casting="no")

        # But is allowed with "equiv":
        res = call_ufunc(arr_bs, casting="equiv")
        assert_array_equal(expected, res)

        # Casting to float64 is safe, but not equiv:
        with pytest.raises(TypeError):
            call_ufunc(arr_bs, dtype=np.float64, casting="equiv")

        # but it is safe cast:
        res = call_ufunc(arr_bs, dtype=np.float64, casting="safe")
        expected = call_ufunc(arr.astype(np.float64))  # upcast
        assert_array_equal(expected, res)

    def test_true_divide(self):
        a = np.array(10)
        b = np.array(20)
        tgt = np.array(0.5)

        for tc in 'bhilqBHILQefdgFDG':
            dt = np.dtype(tc)
            aa = a.astype(dt)
            bb = b.astype(dt)

            # Check result value and dtype.
            for x, y in itertools.product([aa, -aa], [bb, -bb]):

                # Check with no output type specified
                if tc in 'FDG':
                    tgt = complex(x)/complex(y)
                else:
                    tgt = float(x)/float(y)

                res = np.true_divide(x, y)
                rtol = max(np.finfo(res).resolution, 1e-15)
                assert_allclose(res, tgt, rtol=rtol)

                if tc in 'bhilqBHILQ':
                    assert_(res.dtype.name == 'float64')
                else:
                    assert_(res.dtype.name == dt.name )

                # Check with output type specified.  This also checks for the
                # incorrect casts in issue gh-3484 because the unary '-' does
                # not change types, even for unsigned types, Hence casts in the
                # ufunc from signed to unsigned and vice versa will lead to
                # errors in the values.
                for tcout in 'bhilqBHILQ':
                    dtout = np.dtype(tcout)
                    assert_raises(TypeError, np.true_divide, x, y, dtype=dtout)

                for tcout in 'efdg':
                    dtout = np.dtype(tcout)
                    if tc in 'FDG':
                        # Casting complex to float is not allowed
                        assert_raises(TypeError, np.true_divide, x, y, dtype=dtout)
                    else:
                        tgt = float(x)/float(y)
                        rtol = max(np.finfo(dtout).resolution, 1e-15)
                        # The value of tiny for double double is NaN
                        with suppress_warnings() as sup:
                            sup.filter(UserWarning)
                            if not np.isnan(np.finfo(dtout).tiny):
                                atol = max(np.finfo(dtout).tiny, 3e-308)
                            else:
                                atol = 3e-308
                        # Some test values result in invalid for float16.
                        with np.errstate(invalid='ignore'):
                            res = np.true_divide(x, y, dtype=dtout)
                        if not np.isfinite(res) and tcout == 'e':
                            continue
                        assert_allclose(res, tgt, rtol=rtol, atol=atol)
                        assert_(res.dtype.name == dtout.name)

                for tcout in 'FDG':
                    dtout = np.dtype(tcout)
                    tgt = complex(x)/complex(y)
                    rtol = max(np.finfo(dtout).resolution, 1e-15)
                    # The value of tiny for double double is NaN
                    with suppress_warnings() as sup:
                        sup.filter(UserWarning)
                        if not np.isnan(np.finfo(dtout).tiny):
                            atol = max(np.finfo(dtout).tiny, 3e-308)
                        else:
                            atol = 3e-308
                    res = np.true_divide(x, y, dtype=dtout)
                    if not np.isfinite(res):
                        continue
                    assert_allclose(res, tgt, rtol=rtol, atol=atol)
                    assert_(res.dtype.name == dtout.name)

        # Check booleans
        a = np.ones((), dtype=np.bool_)
        res = np.true_divide(a, a)
        assert_(res == 1.0)
        assert_(res.dtype.name == 'float64')
        res = np.true_divide(~a, a)
        assert_(res == 0.0)
        assert_(res.dtype.name == 'float64')

    def test_sum_stability(self):
        a = np.ones(500, dtype=np.float32)
        assert_almost_equal((a / 10.).sum() - a.size / 10., 0, 4)

        a = np.ones(500, dtype=np.float64)
        assert_almost_equal((a / 10.).sum() - a.size / 10., 0, 13)

    def test_sum(self):
        for dt in (int, np.float16, np.float32, np.float64, np.longdouble):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127,
                      128, 1024, 1235):
                tgt = dt(v * (v + 1) / 2)
                d = np.arange(1, v + 1, dtype=dt)

                # warning if sum overflows, which it does in float16
                overflow = not np.isfinite(tgt)

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    assert_almost_equal(np.sum(d), tgt)
                    assert_equal(len(w), 1 * overflow)

                    assert_almost_equal(np.sum(d[::-1]), tgt)
                    assert_equal(len(w), 2 * overflow)

            d = np.ones(500, dtype=dt)
            assert_almost_equal(np.sum(d[::2]), 250.)
            assert_almost_equal(np.sum(d[1::2]), 250.)
            assert_almost_equal(np.sum(d[::3]), 167.)
            assert_almost_equal(np.sum(d[1::3]), 167.)
            assert_almost_equal(np.sum(d[::-2]), 250.)
            assert_almost_equal(np.sum(d[-1::-2]), 250.)
            assert_almost_equal(np.sum(d[::-3]), 167.)
            assert_almost_equal(np.sum(d[-1::-3]), 167.)
            # sum with first reduction entry != 0
            d = np.ones((1,), dtype=dt)
            d += d
            assert_almost_equal(d, 2.)

    def test_sum_complex(self):
        for dt in (np.complex64, np.complex128, np.clongdouble):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127,
                      128, 1024, 1235):
                tgt = dt(v * (v + 1) / 2) - dt((v * (v + 1) / 2) * 1j)
                d = np.empty(v, dtype=dt)
                d.real = np.arange(1, v + 1)
                d.imag = -np.arange(1, v + 1)
                assert_almost_equal(np.sum(d), tgt)
                assert_almost_equal(np.sum(d[::-1]), tgt)

            d = np.ones(500, dtype=dt) + 1j
            assert_almost_equal(np.sum(d[::2]), 250. + 250j)
            assert_almost_equal(np.sum(d[1::2]), 250. + 250j)
            assert_almost_equal(np.sum(d[::3]), 167. + 167j)
            assert_almost_equal(np.sum(d[1::3]), 167. + 167j)
            assert_almost_equal(np.sum(d[::-2]), 250. + 250j)
            assert_almost_equal(np.sum(d[-1::-2]), 250. + 250j)
            assert_almost_equal(np.sum(d[::-3]), 167. + 167j)
            assert_almost_equal(np.sum(d[-1::-3]), 167. + 167j)
            # sum with first reduction entry != 0
            d = np.ones((1,), dtype=dt) + 1j
            d += d
            assert_almost_equal(d, 2. + 2j)

    def test_sum_initial(self):
        # Integer, single axis
        assert_equal(np.sum([3], initial=2), 5)

        # Floating point
        assert_almost_equal(np.sum([0.2], initial=0.1), 0.3)

        # Multiple non-adjacent axes
        assert_equal(np.sum(np.ones((2, 3, 5), dtype=np.int64), axis=(0, 2), initial=2),
                     [12, 12, 12])

    def test_sum_where(self):
        # More extensive tests done in test_reduction_with_where.
        assert_equal(np.sum([[1., 2.], [3., 4.]], where=[True, False]), 4.)
        assert_equal(np.sum([[1., 2.], [3., 4.]], axis=0, initial=5.,
                            where=[True, False]), [9., 5.])

    def test_inner1d(self):
        a = np.arange(6).reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a*a, axis=-1))
        a = np.arange(6)
        assert_array_equal(umt.inner1d(a, a), np.sum(a*a))

    def test_broadcast(self):
        msg = "broadcast"
        a = np.arange(4).reshape((2, 1, 2))
        b = np.arange(4).reshape((1, 2, 2))
        assert_array_equal(umt.inner1d(a, b), np.sum(a*b, axis=-1), err_msg=msg)
        msg = "extend & broadcast loop dimensions"
        b = np.arange(4).reshape((2, 2))
        assert_array_equal(umt.inner1d(a, b), np.sum(a*b, axis=-1), err_msg=msg)
        # Broadcast in core dimensions should fail
        a = np.arange(8).reshape((4, 2))
        b = np.arange(4).reshape((4, 1))
        assert_raises(ValueError, umt.inner1d, a, b)
        # Extend core dimensions should fail
        a = np.arange(8).reshape((4, 2))
        b = np.array(7)
        assert_raises(ValueError, umt.inner1d, a, b)
        # Broadcast should fail
        a = np.arange(2).reshape((2, 1, 1))
        b = np.arange(3).reshape((3, 1, 1))
        assert_raises(ValueError, umt.inner1d, a, b)

        # Writing to a broadcasted array with overlap should warn, gh-2705
        a = np.arange(2)
        b = np.arange(4).reshape((2, 2))
        u, v = np.broadcast_arrays(a, b)
        assert_equal(u.strides[0], 0)
        x = u + v
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            u += v
            assert_equal(len(w), 1)
            assert_(x[0, 0] != u[0, 0])

        # Output reduction should not be allowed.
        # See gh-15139
        a = np.arange(6).reshape(3, 2)
        b = np.ones(2)
        out = np.empty(())
        assert_raises(ValueError, umt.inner1d, a, b, out)
        out2 = np.empty(3)
        c = umt.inner1d(a, b, out2)
        assert_(c is out2)

    def test_out_broadcasts(self):
        # For ufuncs and gufuncs (not for reductions), we currently allow
        # the output to cause broadcasting of the input arrays.
        # both along dimensions with shape 1 and dimensions which do not
        # exist at all in the inputs.
        arr = np.arange(3).reshape(1, 3)
        out = np.empty((5, 4, 3))
        np.add(arr, arr, out=out)
        assert (out == np.arange(3) * 2).all()

        # The same holds for gufuncs (gh-16484)
        umt.inner1d(arr, arr, out=out)
        # the result would be just a scalar `5`, but is broadcast fully:
        assert (out == 5).all()

    def test_type_cast(self):
        msg = "type cast"
        a = np.arange(6, dtype='short').reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a*a, axis=-1),
                           err_msg=msg)
        msg = "type cast on one argument"
        a = np.arange(6).reshape((2, 3))
        b = a + 0.1
        assert_array_almost_equal(umt.inner1d(a, b), np.sum(a*b, axis=-1),
                                  err_msg=msg)

    def test_endian(self):
        msg = "big endian"
        a = np.arange(6, dtype='>i4').reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a*a, axis=-1),
                           err_msg=msg)
        msg = "little endian"
        a = np.arange(6, dtype='<i4').reshape((2, 3))
        assert_array_equal(umt.inner1d(a, a), np.sum(a*a, axis=-1),
                           err_msg=msg)

        # Output should always be native-endian
        Ba = np.arange(1, dtype='>f8')
        La = np.arange(1, dtype='<f8')
        assert_equal((Ba+Ba).dtype, np.dtype('f8'))
        assert_equal((Ba+La).dtype, np.dtype('f8'))
        assert_equal((La+Ba).dtype, np.dtype('f8'))
        assert_equal((La+La).dtype, np.dtype('f8'))

        assert_equal(np.absolute(La).dtype, np.dtype('f8'))
        assert_equal(np.absolute(Ba).dtype, np.dtype('f8'))
        assert_equal(np.negative(La).dtype, np.dtype('f8'))
        assert_equal(np.negative(Ba).dtype, np.dtype('f8'))

    def test_incontiguous_array(self):
        msg = "incontiguous memory layout of array"
        x = np.arange(64).reshape((2, 2, 2, 2, 2, 2))
        a = x[:, 0,:, 0,:, 0]
        b = x[:, 1,:, 1,:, 1]
        a[0, 0, 0] = -1
        msg2 = "make sure it references to the original array"
        assert_equal(x[0, 0, 0, 0, 0, 0], -1, err_msg=msg2)
        assert_array_equal(umt.inner1d(a, b), np.sum(a*b, axis=-1), err_msg=msg)
        x = np.arange(24).reshape(2, 3, 4)
        a = x.T
        b = x.T
        a[0, 0, 0] = -1
        assert_equal(x[0, 0, 0], -1, err_msg=msg2)
        assert_array_equal(umt.inner1d(a, b), np.sum(a*b, axis=-1), err_msg=msg)

    def test_output_argument(self):
        msg = "output argument"
        a = np.arange(12).reshape((2, 3, 2))
        b = np.arange(4).reshape((2, 1, 2)) + 1
        c = np.zeros((2, 3), dtype='int')
        umt.inner1d(a, b, c)
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a, b, out=c)
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)

        msg = "output argument with type cast"
        c = np.zeros((2, 3), dtype='int16')
        umt.inner1d(a, b, c)
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a, b, out=c)
        assert_array_equal(c, np.sum(a*b, axis=-1), err_msg=msg)

        msg = "output argument with incontiguous layout"
        c = np.zeros((2, 3, 4), dtype='int16')
        umt.inner1d(a, b, c[..., 0])
        assert_array_equal(c[..., 0], np.sum(a*b, axis=-1), err_msg=msg)
        c[:] = -1
        umt.inner1d(a, b, out=c[..., 0])
        assert_array_equal(c[..., 0], np.sum(a*b, axis=-1), err_msg=msg)

    def test_axes_argument(self):
        # inner1d signature: '(i),(i)->()'
        inner1d = umt.inner1d
        a = np.arange(27.).reshape((3, 3, 3))
        b = np.arange(10., 19.).reshape((3, 1, 3))
        # basic tests on inputs (outputs tested below with matrix_multiply).
        c = inner1d(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        # default
        c = inner1d(a, b, axes=[(-1,), (-1,), ()])
        assert_array_equal(c, (a * b).sum(-1))
        # integers ok for single axis.
        c = inner1d(a, b, axes=[-1, -1, ()])
        assert_array_equal(c, (a * b).sum(-1))
        # mix fine
        c = inner1d(a, b, axes=[(-1,), -1, ()])
        assert_array_equal(c, (a * b).sum(-1))
        # can omit last axis.
        c = inner1d(a, b, axes=[-1, -1])
        assert_array_equal(c, (a * b).sum(-1))
        # can pass in other types of integer (with __index__ protocol)
        c = inner1d(a, b, axes=[np.int8(-1), np.array(-1, dtype=np.int32)])
        assert_array_equal(c, (a * b).sum(-1))
        # swap some axes
        c = inner1d(a, b, axes=[0, 0])
        assert_array_equal(c, (a * b).sum(0))
        c = inner1d(a, b, axes=[0, 2])
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1))
        # Check errors for improperly constructed axes arguments.
        # should have list.
        assert_raises(TypeError, inner1d, a, b, axes=-1)
        # needs enough elements
        assert_raises(ValueError, inner1d, a, b, axes=[-1])
        # should pass in indices.
        assert_raises(TypeError, inner1d, a, b, axes=[-1.0, -1.0])
        assert_raises(TypeError, inner1d, a, b, axes=[(-1.0,), -1])
        assert_raises(TypeError, inner1d, a, b, axes=[None, 1])
        # cannot pass an index unless there is only one dimension
        # (output is wrong in this case)
        assert_raises(TypeError, inner1d, a, b, axes=[-1, -1, -1])
        # or pass in generally the wrong number of axes
        assert_raises(ValueError, inner1d, a, b, axes=[-1, -1, (-1,)])
        assert_raises(ValueError, inner1d, a, b, axes=[-1, (-2, -1), ()])
        # axes need to have same length.
        assert_raises(ValueError, inner1d, a, b, axes=[0, 1])

        # matrix_multiply signature: '(m,n),(n,p)->(m,p)'
        mm = umt.matrix_multiply
        a = np.arange(12).reshape((2, 3, 2))
        b = np.arange(8).reshape((2, 2, 2, 1)) + 1
        # Sanity check.
        c = mm(a, b)
        assert_array_equal(c, np.matmul(a, b))
        # Default axes.
        c = mm(a, b, axes=[(-2, -1), (-2, -1), (-2, -1)])
        assert_array_equal(c, np.matmul(a, b))
        # Default with explicit axes.
        c = mm(a, b, axes=[(1, 2), (2, 3), (2, 3)])
        assert_array_equal(c, np.matmul(a, b))
        # swap some axes.
        c = mm(a, b, axes=[(0, -1), (1, 2), (-2, -1)])
        assert_array_equal(c, np.matmul(a.transpose(1, 0, 2),
                                        b.transpose(0, 3, 1, 2)))
        # Default with output array.
        c = np.empty((2, 2, 3, 1))
        d = mm(a, b, out=c, axes=[(1, 2), (2, 3), (2, 3)])
        assert_(c is d)
        assert_array_equal(c, np.matmul(a, b))
        # Transposed output array
        c = np.empty((1, 2, 2, 3))
        d = mm(a, b, out=c, axes=[(-2, -1), (-2, -1), (3, 0)])
        assert_(c is d)
        assert_array_equal(c, np.matmul(a, b).transpose(3, 0, 1, 2))
        # Check errors for improperly constructed axes arguments.
        # wrong argument
        assert_raises(TypeError, mm, a, b, axis=1)
        # axes should be list
        assert_raises(TypeError, mm, a, b, axes=1)
        assert_raises(TypeError, mm, a, b, axes=((-2, -1), (-2, -1), (-2, -1)))
        # list needs to have right length
        assert_raises(ValueError, mm, a, b, axes=[])
        assert_raises(ValueError, mm, a, b, axes=[(-2, -1)])
        # list should contain tuples for multiple axes
        assert_raises(TypeError, mm, a, b, axes=[-1, -1, -1])
        assert_raises(TypeError, mm, a, b, axes=[(-2, -1), (-2, -1), -1])
        assert_raises(TypeError,
                      mm, a, b, axes=[[-2, -1], [-2, -1], [-2, -1]])
        assert_raises(TypeError,
                      mm, a, b, axes=[(-2, -1), (-2, -1), [-2, -1]])
        assert_raises(TypeError, mm, a, b, axes=[(-2, -1), (-2, -1), None])
        # tuples should not have duplicated values
        assert_raises(ValueError, mm, a, b, axes=[(-2, -1), (-2, -1), (-2, -2)])
        # arrays should have enough axes.
        z = np.zeros((2, 2))
        assert_raises(ValueError, mm, z, z[0])
        assert_raises(ValueError, mm, z, z, out=z[:, 0])
        assert_raises(ValueError, mm, z[1], z, axes=[0, 1])
        assert_raises(ValueError, mm, z, z, out=z[0], axes=[0, 1])
        # Regular ufuncs should not accept axes.
        assert_raises(TypeError, np.add, 1., 1., axes=[0])
        # should be able to deal with bad unrelated kwargs.
        assert_raises(TypeError, mm, z, z, axes=[0, 1], parrot=True)

    def test_axis_argument(self):
        # inner1d signature: '(i),(i)->()'
        inner1d = umt.inner1d
        a = np.arange(27.).reshape((3, 3, 3))
        b = np.arange(10., 19.).reshape((3, 1, 3))
        c = inner1d(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axis=-1)
        assert_array_equal(c, (a * b).sum(-1))
        out = np.zeros_like(c)
        d = inner1d(a, b, axis=-1, out=out)
        assert_(d is out)
        assert_array_equal(d, c)
        c = inner1d(a, b, axis=0)
        assert_array_equal(c, (a * b).sum(0))
        # Sanity checks on innerwt and cumsum.
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w, axis=0),
                           np.sum(a * b * w, axis=0))
        assert_array_equal(umt.cumsum(a, axis=0), np.cumsum(a, axis=0))
        assert_array_equal(umt.cumsum(a, axis=-1), np.cumsum(a, axis=-1))
        out = np.empty_like(a)
        b = umt.cumsum(a, out=out, axis=0)
        assert_(out is b)
        assert_array_equal(b, np.cumsum(a, axis=0))
        b = umt.cumsum(a, out=out, axis=1)
        assert_(out is b)
        assert_array_equal(b, np.cumsum(a, axis=-1))
        # Check errors.
        # Cannot pass in both axis and axes.
        assert_raises(TypeError, inner1d, a, b, axis=0, axes=[0, 0])
        # Not an integer.
        assert_raises(TypeError, inner1d, a, b, axis=[0])
        # more than 1 core dimensions.
        mm = umt.matrix_multiply
        assert_raises(TypeError, mm, a, b, axis=1)
        # Output wrong size in axis.
        out = np.empty((1, 2, 3), dtype=a.dtype)
        assert_raises(ValueError, umt.cumsum, a, out=out, axis=0)
        # Regular ufuncs should not accept axis.
        assert_raises(TypeError, np.add, 1., 1., axis=0)

    def test_keepdims_argument(self):
        # inner1d signature: '(i),(i)->()'
        inner1d = umt.inner1d
        a = np.arange(27.).reshape((3, 3, 3))
        b = np.arange(10., 19.).reshape((3, 1, 3))
        c = inner1d(a, b)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, keepdims=False)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        out = np.zeros_like(c)
        d = inner1d(a, b, keepdims=True, out=out)
        assert_(d is out)
        assert_array_equal(d, c)
        # Now combined with axis and axes.
        c = inner1d(a, b, axis=-1, keepdims=False)
        assert_array_equal(c, (a * b).sum(-1, keepdims=False))
        c = inner1d(a, b, axis=-1, keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        c = inner1d(a, b, axis=0, keepdims=False)
        assert_array_equal(c, (a * b).sum(0, keepdims=False))
        c = inner1d(a, b, axis=0, keepdims=True)
        assert_array_equal(c, (a * b).sum(0, keepdims=True))
        c = inner1d(a, b, axes=[(-1,), (-1,), ()], keepdims=False)
        assert_array_equal(c, (a * b).sum(-1))
        c = inner1d(a, b, axes=[(-1,), (-1,), (-1,)], keepdims=True)
        assert_array_equal(c, (a * b).sum(-1, keepdims=True))
        c = inner1d(a, b, axes=[0, 0], keepdims=False)
        assert_array_equal(c, (a * b).sum(0))
        c = inner1d(a, b, axes=[0, 0, 0], keepdims=True)
        assert_array_equal(c, (a * b).sum(0, keepdims=True))
        c = inner1d(a, b, axes=[0, 2], keepdims=False)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1))
        c = inner1d(a, b, axes=[0, 2], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1,
                                                             keepdims=True))
        c = inner1d(a, b, axes=[0, 2, 2], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 2, 0) * b).sum(-1,
                                                             keepdims=True))
        c = inner1d(a, b, axes=[0, 2, 0], keepdims=True)
        assert_array_equal(c, (a * b.transpose(2, 0, 1)).sum(0, keepdims=True))
        # Hardly useful, but should work.
        c = inner1d(a, b, axes=[0, 2, 1], keepdims=True)
        assert_array_equal(c, (a.transpose(1, 0, 2) * b.transpose(0, 2, 1))
                           .sum(1, keepdims=True))
        # Check with two core dimensions.
        a = np.eye(3) * np.arange(4.)[:, np.newaxis, np.newaxis]
        expected = uml.det(a)
        c = uml.det(a, keepdims=False)
        assert_array_equal(c, expected)
        c = uml.det(a, keepdims=True)
        assert_array_equal(c, expected[:, np.newaxis, np.newaxis])
        a = np.eye(3) * np.arange(4.)[:, np.newaxis, np.newaxis]
        expected_s, expected_l = uml.slogdet(a)
        cs, cl = uml.slogdet(a, keepdims=False)
        assert_array_equal(cs, expected_s)
        assert_array_equal(cl, expected_l)
        cs, cl = uml.slogdet(a, keepdims=True)
        assert_array_equal(cs, expected_s[:, np.newaxis, np.newaxis])
        assert_array_equal(cl, expected_l[:, np.newaxis, np.newaxis])
        # Sanity check on innerwt.
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w, keepdims=True),
                           np.sum(a * b * w, axis=-1, keepdims=True))
        assert_array_equal(umt.innerwt(a, b, w, axis=0, keepdims=True),
                           np.sum(a * b * w, axis=0, keepdims=True))
        # Check errors.
        # Not a boolean
        assert_raises(TypeError, inner1d, a, b, keepdims='true')
        # More than 1 core dimension, and core output dimensions.
        mm = umt.matrix_multiply
        assert_raises(TypeError, mm, a, b, keepdims=True)
        assert_raises(TypeError, mm, a, b, keepdims=False)
        # Regular ufuncs should not accept keepdims.
        assert_raises(TypeError, np.add, 1., 1., keepdims=False)

    def test_innerwt(self):
        a = np.arange(6).reshape((2, 3))
        b = np.arange(10, 16).reshape((2, 3))
        w = np.arange(20, 26).reshape((2, 3))
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a*b*w, axis=-1))
        a = np.arange(100, 124).reshape((2, 3, 4))
        b = np.arange(200, 224).reshape((2, 3, 4))
        w = np.arange(300, 324).reshape((2, 3, 4))
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a*b*w, axis=-1))

    def test_innerwt_empty(self):
        """Test generalized ufunc with zero-sized operands"""
        a = np.array([], dtype='f8')
        b = np.array([], dtype='f8')
        w = np.array([], dtype='f8')
        assert_array_equal(umt.innerwt(a, b, w), np.sum(a*b*w, axis=-1))

    def test_cross1d(self):
        """Test with fixed-sized signature."""
        a = np.eye(3)
        assert_array_equal(umt.cross1d(a, a), np.zeros((3, 3)))
        out = np.zeros((3, 3))
        result = umt.cross1d(a[0], a, out)
        assert_(result is out)
        assert_array_equal(result, np.vstack((np.zeros(3), a[2], -a[1])))
        assert_raises(ValueError, umt.cross1d, np.eye(4), np.eye(4))
        assert_raises(ValueError, umt.cross1d, a, np.arange(4.))
        # Wrong output core dimension.
        assert_raises(ValueError, umt.cross1d, a, np.arange(3.), np.zeros((3, 4)))
        # Wrong output broadcast dimension (see gh-15139).
        assert_raises(ValueError, umt.cross1d, a, np.arange(3.), np.zeros(3))

    def test_can_ignore_signature(self):
        # Comparing the effects of ? in signature:
        # matrix_multiply: (m,n),(n,p)->(m,p)    # all must be there.
        # matmul:        (m?,n),(n,p?)->(m?,p?)  # allow missing m, p.
        mat = np.arange(12).reshape((2, 3, 2))
        single_vec = np.arange(2)
        col_vec = single_vec[:, np.newaxis]
        col_vec_array = np.arange(8).reshape((2, 2, 2, 1)) + 1
        # matrix @ single column vector with proper dimension
        mm_col_vec = umt.matrix_multiply(mat, col_vec)
        # matmul does the same thing
        matmul_col_vec = umt.matmul(mat, col_vec)
        assert_array_equal(matmul_col_vec, mm_col_vec)
        # matrix @ vector without dimension making it a column vector.
        # matrix multiply fails -> missing core dim.
        assert_raises(ValueError, umt.matrix_multiply, mat, single_vec)
        # matmul mimicker passes, and returns a vector.
        matmul_col = umt.matmul(mat, single_vec)
        assert_array_equal(matmul_col, mm_col_vec.squeeze())
        # Now with a column array: same as for column vector,
        # broadcasting sensibly.
        mm_col_vec = umt.matrix_multiply(mat, col_vec_array)
        matmul_col_vec = umt.matmul(mat, col_vec_array)
        assert_array_equal(matmul_col_vec, mm_col_vec)
        # As above, but for row vector
        single_vec = np.arange(3)
        row_vec = single_vec[np.newaxis, :]
        row_vec_array = np.arange(24).reshape((4, 2, 1, 1, 3)) + 1
        # row vector @ matrix
        mm_row_vec = umt.matrix_multiply(row_vec, mat)
        matmul_row_vec = umt.matmul(row_vec, mat)
        assert_array_equal(matmul_row_vec, mm_row_vec)
        # single row vector @ matrix
        assert_raises(ValueError, umt.matrix_multiply, single_vec, mat)
        matmul_row = umt.matmul(single_vec, mat)
        assert_array_equal(matmul_row, mm_row_vec.squeeze())
        # row vector array @ matrix
        mm_row_vec = umt.matrix_multiply(row_vec_array, mat)
        matmul_row_vec = umt.matmul(row_vec_array, mat)
        assert_array_equal(matmul_row_vec, mm_row_vec)
        # Now for vector combinations
        # row vector @ column vector
        col_vec = row_vec.T
        col_vec_array = row_vec_array.swapaxes(-2, -1)
        mm_row_col_vec = umt.matrix_multiply(row_vec, col_vec)
        matmul_row_col_vec = umt.matmul(row_vec, col_vec)
        assert_array_equal(matmul_row_col_vec, mm_row_col_vec)
        # single row vector @ single col vector
        assert_raises(ValueError, umt.matrix_multiply, single_vec, single_vec)
        matmul_row_col = umt.matmul(single_vec, single_vec)
        assert_array_equal(matmul_row_col, mm_row_col_vec.squeeze())
        # row vector array @ matrix
        mm_row_col_array = umt.matrix_multiply(row_vec_array, col_vec_array)
        matmul_row_col_array = umt.matmul(row_vec_array, col_vec_array)
        assert_array_equal(matmul_row_col_array, mm_row_col_array)
        # Finally, check that things are *not* squeezed if one gives an
        # output.
        out = np.zeros_like(mm_row_col_array)
        out = umt.matrix_multiply(row_vec_array, col_vec_array, out=out)
        assert_array_equal(out, mm_row_col_array)
        out[:] = 0
        out = umt.matmul(row_vec_array, col_vec_array, out=out)
        assert_array_equal(out, mm_row_col_array)
        # And check one cannot put missing dimensions back.
        out = np.zeros_like(mm_row_col_vec)
        assert_raises(ValueError, umt.matrix_multiply, single_vec, single_vec,
                      out)
        # But fine for matmul, since it is just a broadcast.
        out = umt.matmul(single_vec, single_vec, out)
        assert_array_equal(out, mm_row_col_vec.squeeze())

    def test_matrix_multiply(self):
        self.compare_matrix_multiply_results(np.int64)
        self.compare_matrix_multiply_results(np.double)

    def test_matrix_multiply_umath_empty(self):
        res = umt.matrix_multiply(np.ones((0, 10)), np.ones((10, 0)))
        assert_array_equal(res, np.zeros((0, 0)))
        res = umt.matrix_multiply(np.ones((10, 0)), np.ones((0, 10)))
        assert_array_equal(res, np.zeros((10, 10)))

    def compare_matrix_multiply_results(self, tp):
        d1 = np.array(np.random.rand(2, 3, 4), dtype=tp)
        d2 = np.array(np.random.rand(2, 3, 4), dtype=tp)
        msg = "matrix multiply on type %s" % d1.dtype.name

        def permute_n(n):
            if n == 1:
                return ([0],)
            ret = ()
            base = permute_n(n-1)
            for perm in base:
                for i in range(n):
                    new = perm + [n-1]
                    new[n-1] = new[i]
                    new[i] = n-1
                    ret += (new,)
            return ret

        def slice_n(n):
            if n == 0:
                return ((),)
            ret = ()
            base = slice_n(n-1)
            for sl in base:
                ret += (sl+(slice(None),),)
                ret += (sl+(slice(0, 1),),)
            return ret

        def broadcastable(s1, s2):
            return s1 == s2 or s1 == 1 or s2 == 1

        permute_3 = permute_n(3)
        slice_3 = slice_n(3) + ((slice(None, None, -1),)*3,)

        ref = True
        for p1 in permute_3:
            for p2 in permute_3:
                for s1 in slice_3:
                    for s2 in slice_3:
                        a1 = d1.transpose(p1)[s1]
                        a2 = d2.transpose(p2)[s2]
                        ref = ref and a1.base is not None
                        ref = ref and a2.base is not None
                        if (a1.shape[-1] == a2.shape[-2] and
                                broadcastable(a1.shape[0], a2.shape[0])):
                            assert_array_almost_equal(
                                umt.matrix_multiply(a1, a2),
                                np.sum(a2[..., np.newaxis].swapaxes(-3, -1) *
                                       a1[..., np.newaxis,:], axis=-1),
                                err_msg=msg + ' %s %s' % (str(a1.shape),
                                                          str(a2.shape)))

        assert_equal(ref, True, err_msg="reference check")

    def test_euclidean_pdist(self):
        a = np.arange(12, dtype=float).reshape(4, 3)
        out = np.empty((a.shape[0] * (a.shape[0] - 1) // 2,), dtype=a.dtype)
        umt.euclidean_pdist(a, out)
        b = np.sqrt(np.sum((a[:, None] - a)**2, axis=-1))
        b = b[~np.tri(a.shape[0], dtype=bool)]
        assert_almost_equal(out, b)
        # An output array is required to determine p with signature (n,d)->(p)
        assert_raises(ValueError, umt.euclidean_pdist, a)

    def test_cumsum(self):
        a = np.arange(10)
        result = umt.cumsum(a)
        assert_array_equal(result, a.cumsum())

    def test_object_logical(self):
        a = np.array([3, None, True, False, "test", ""], dtype=object)
        assert_equal(np.logical_or(a, None),
                        np.array([x or None for x in a], dtype=object))
        assert_equal(np.logical_or(a, True),
                        np.array([x or True for x in a], dtype=object))
        assert_equal(np.logical_or(a, 12),
                        np.array([x or 12 for x in a], dtype=object))
        assert_equal(np.logical_or(a, "blah"),
                        np.array([x or "blah" for x in a], dtype=object))

        assert_equal(np.logical_and(a, None),
                        np.array([x and None for x in a], dtype=object))
        assert_equal(np.logical_and(a, True),
                        np.array([x and True for x in a], dtype=object))
        assert_equal(np.logical_and(a, 12),
                        np.array([x and 12 for x in a], dtype=object))
        assert_equal(np.logical_and(a, "blah"),
                        np.array([x and "blah" for x in a], dtype=object))

        assert_equal(np.logical_not(a),
                        np.array([not x for x in a], dtype=object))

        assert_equal(np.logical_or.reduce(a), 3)
        assert_equal(np.logical_and.reduce(a), None)

    def test_object_comparison(self):
        class HasComparisons:
            def __eq__(self, other):
                return '=='

        arr0d = np.array(HasComparisons())
        assert_equal(arr0d == arr0d, True)
        assert_equal(np.equal(arr0d, arr0d), True)  # normal behavior is a cast

        arr1d = np.array([HasComparisons()])
        assert_equal(arr1d == arr1d, np.array([True]))
        assert_equal(np.equal(arr1d, arr1d), np.array([True]))  # normal behavior is a cast
        assert_equal(np.equal(arr1d, arr1d, dtype=object), np.array(['==']))

    def test_object_array_reduction(self):
        # Reductions on object arrays
        a = np.array(['a', 'b', 'c'], dtype=object)
        assert_equal(np.sum(a), 'abc')
        assert_equal(np.max(a), 'c')
        assert_equal(np.min(a), 'a')
        a = np.array([True, False, True], dtype=object)
        assert_equal(np.sum(a), 2)
        assert_equal(np.prod(a), 0)
        assert_equal(np.any(a), True)
        assert_equal(np.all(a), False)
        assert_equal(np.max(a), True)
        assert_equal(np.min(a), False)
        assert_equal(np.array([[1]], dtype=object).sum(), 1)
        assert_equal(np.array([[[1, 2]]], dtype=object).sum((0, 1)), [1, 2])
        assert_equal(np.array([1], dtype=object).sum(initial=1), 2)
        assert_equal(np.array([[1], [2, 3]], dtype=object)
                     .sum(initial=[0], where=[False, True]), [0, 2, 3])

    def test_object_array_accumulate_inplace(self):
        # Checks that in-place accumulates work, see also gh-7402
        arr = np.ones(4, dtype=object)
        arr[:] = [[1] for i in range(4)]
        # Twice reproduced also for tuples:
        np.add.accumulate(arr, out=arr)
        np.add.accumulate(arr, out=arr)
        assert_array_equal(arr,
                           np.array([[1]*i for i in [1, 3, 6, 10]], dtype=object),
                          )

        # And the same if the axis argument is used
        arr = np.ones((2, 4), dtype=object)
        arr[0, :] = [[2] for i in range(4)]
        np.add.accumulate(arr, out=arr, axis=-1)
        np.add.accumulate(arr, out=arr, axis=-1)
        assert_array_equal(arr[0, :],
                           np.array([[2]*i for i in [1, 3, 6, 10]], dtype=object),
                          )

    def test_object_array_accumulate_failure(self):
        # Typical accumulation on object works as expected:
        res = np.add.accumulate(np.array([1, 0, 2], dtype=object))
        assert_array_equal(res, np.array([1, 1, 3], dtype=object))
        # But errors are propagated from the inner-loop if they occur:
        with pytest.raises(TypeError):
            np.add.accumulate([1, None, 2])

    def test_object_array_reduceat_inplace(self):
        # Checks that in-place reduceats work, see also gh-7465
        arr = np.empty(4, dtype=object)
        arr[:] = [[1] for i in range(4)]
        out = np.empty(4, dtype=object)
        out[:] = [[1] for i in range(4)]
        np.add.reduceat(arr, np.arange(4), out=arr)
        np.add.reduceat(arr, np.arange(4), out=arr)
        assert_array_equal(arr, out)

        # And the same if the axis argument is used
        arr = np.ones((2, 4), dtype=object)
        arr[0, :] = [[2] for i in range(4)]
        out = np.ones((2, 4), dtype=object)
        out[0, :] = [[2] for i in range(4)]
        np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
        np.add.reduceat(arr, np.arange(4), out=arr, axis=-1)
        assert_array_equal(arr, out)

    def test_object_array_reduceat_failure(self):
        # Reduceat works as expected when no invalid operation occurs (None is
        # not involved in an operation here)
        res = np.add.reduceat(np.array([1, None, 2], dtype=object), [1, 2])
        assert_array_equal(res, np.array([None, 2], dtype=object))
        # But errors when None would be involved in an operation:
        with pytest.raises(TypeError):
            np.add.reduceat([1, None, 2], [0, 2])

    def test_zerosize_reduction(self):
        # Test with default dtype and object dtype
        for a in [[], np.array([], dtype=object)]:
            assert_equal(np.sum(a), 0)
            assert_equal(np.prod(a), 1)
            assert_equal(np.any(a), False)
            assert_equal(np.all(a), True)
            assert_raises(ValueError, np.max, a)
            assert_raises(ValueError, np.min, a)

    def test_axis_out_of_bounds(self):
        a = np.array([False, False])
        assert_raises(np.AxisError, a.all, axis=1)
        a = np.array([False, False])
        assert_raises(np.AxisError, a.all, axis=-2)

        a = np.array([False, False])
        assert_raises(np.AxisError, a.any, axis=1)
        a = np.array([False, False])
        assert_raises(np.AxisError, a.any, axis=-2)

    def test_scalar_reduction(self):
        # The functions 'sum', 'prod', etc allow specifying axis=0
        # even for scalars
        assert_equal(np.sum(3, axis=0), 3)
        assert_equal(np.prod(3.5, axis=0), 3.5)
        assert_equal(np.any(True, axis=0), True)
        assert_equal(np.all(False, axis=0), False)
        assert_equal(np.max(3, axis=0), 3)
        assert_equal(np.min(2.5, axis=0), 2.5)

        # Check scalar behaviour for ufuncs without an identity
        assert_equal(np.power.reduce(3), 3)

        # Make sure that scalars are coming out from this operation
        assert_(type(np.prod(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.sum(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.max(np.float32(2.5), axis=0)) is np.float32)
        assert_(type(np.min(np.float32(2.5), axis=0)) is np.float32)

        # check if scalars/0-d arrays get cast
        assert_(type(np.any(0, axis=0)) is np.bool_)

        # assert that 0-d arrays get wrapped
        class MyArray(np.ndarray):
            pass
        a = np.array(1).view(MyArray)
        assert_(type(np.any(a)) is MyArray)

    def test_casting_out_param(self):
        # Test that it's possible to do casts on output
        a = np.ones((200, 100), np.int64)
        b = np.ones((200, 100), np.int64)
        c = np.ones((200, 100), np.float64)
        np.add(a, b, out=c)
        assert_equal(c, 2)

        a = np.zeros(65536)
        b = np.zeros(65536, dtype=np.float32)
        np.subtract(a, 0, out=b)
        assert_equal(b, 0)

    def test_where_param(self):
        # Test that the where= ufunc parameter works with regular arrays
        a = np.arange(7)
        b = np.ones(7)
        c = np.zeros(7)
        np.add(a, b, out=c, where=(a % 2 == 1))
        assert_equal(c, [0, 2, 0, 4, 0, 6, 0])

        a = np.arange(4).reshape(2, 2) + 2
        np.power(a, [2, 3], out=a, where=[[0, 1], [1, 0]])
        assert_equal(a, [[2, 27], [16, 5]])
        # Broadcasting the where= parameter
        np.subtract(a, 2, out=a, where=[True, False])
        assert_equal(a, [[0, 27], [14, 5]])

    def test_where_param_buffer_output(self):
        # This test is temporarily skipped because it requires
        # adding masking features to the nditer to work properly

        # With casting on output
        a = np.ones(10, np.int64)
        b = np.ones(10, np.int64)
        c = 1.5 * np.ones(10, np.float64)
        np.add(a, b, out=c, where=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
        assert_equal(c, [2, 1.5, 1.5, 2, 1.5, 1.5, 2, 2, 2, 1.5])

    def test_where_param_alloc(self):
        # With casting and allocated output
        a = np.array([1], dtype=np.int64)
        m = np.array([True], dtype=bool)
        assert_equal(np.sqrt(a, where=m), [1])

        # No casting and allocated output
        a = np.array([1], dtype=np.float64)
        m = np.array([True], dtype=bool)
        assert_equal(np.sqrt(a, where=m), [1])

    def test_where_with_broadcasting(self):
        # See gh-17198
        a = np.random.random((5000, 4))
        b = np.random.random((5000, 1))

        where = a > 0.3
        out = np.full_like(a, 0)
        np.less(a, b, where=where, out=out)
        b_where = np.broadcast_to(b, a.shape)[where]
        assert_array_equal((a[where] < b_where), out[where].astype(bool))
        assert not out[~where].any()  # outside mask, out remains all 0

    def check_identityless_reduction(self, a):
        # np.minimum.reduce is an identityless reduction

        # Verify that it sees the zero at various positions
        a[...] = 1
        a[1, 0, 0] = 0
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [0, 1, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [1, 0])
        assert_equal(np.minimum.reduce(a, axis=0),
                                    [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1),
                                    [[1, 1, 1, 1], [0, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2),
                                    [[1, 1, 1], [0, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)

        a[...] = 1
        a[0, 1, 0] = 0
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [0, 1, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [1, 0, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [0, 1])
        assert_equal(np.minimum.reduce(a, axis=0),
                                    [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1),
                                    [[0, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2),
                                    [[1, 0, 1], [1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)

        a[...] = 1
        a[0, 0, 1] = 0
        assert_equal(np.minimum.reduce(a, axis=None), 0)
        assert_equal(np.minimum.reduce(a, axis=(0, 1)), [1, 0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(0, 2)), [0, 1, 1])
        assert_equal(np.minimum.reduce(a, axis=(1, 2)), [0, 1])
        assert_equal(np.minimum.reduce(a, axis=0),
                                    [[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=1),
                                    [[1, 0, 1, 1], [1, 1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=2),
                                    [[0, 1, 1], [1, 1, 1]])
        assert_equal(np.minimum.reduce(a, axis=()), a)

    @requires_memory(6 * 1024**3)
    def test_identityless_reduction_huge_array(self):
        # Regression test for gh-20921 (copying identity incorrectly failed)
        arr = np.zeros((2, 2**31), 'uint8')
        arr[:, 0] = [1, 3]
        arr[:, -1] = [4, 1]
        res = np.maximum.reduce(arr, axis=0)
        del arr
        assert res[0] == 3
        assert res[-1] == 4

    def test_identityless_reduction_corder(self):
        a = np.empty((2, 3, 4), order='C')
        self.check_identityless_reduction(a)

    def test_identityless_reduction_forder(self):
        a = np.empty((2, 3, 4), order='F')
        self.check_identityless_reduction(a)

    def test_identityless_reduction_otherorder(self):
        a = np.empty((2, 4, 3), order='C').swapaxes(1, 2)
        self.check_identityless_reduction(a)

    def test_identityless_reduction_noncontig(self):
        a = np.empty((3, 5, 4), order='C').swapaxes(1, 2)
        a = a[1:, 1:, 1:]
        self.check_identityless_reduction(a)

    def test_identityless_reduction_noncontig_unaligned(self):
        a = np.empty((3*4*5*8 + 1,), dtype='i1')
        a = a[1:].view(dtype='f8')
        a.shape = (3, 4, 5)
        a = a[1:, 1:, 1:]
        self.check_identityless_reduction(a)

    def test_initial_reduction(self):
        # np.minimum.reduce is an identityless reduction

        # For cases like np.maximum(np.abs(...), initial=0)
        # More generally, a supremum over non-negative numbers.
        assert_equal(np.maximum.reduce([], initial=0), 0)

        # For cases like reduction of an empty array over the reals.
        assert_equal(np.minimum.reduce([], initial=np.inf), np.inf)
        assert_equal(np.maximum.reduce([], initial=-np.inf), -np.inf)

        # Random tests
        assert_equal(np.minimum.reduce([5], initial=4), 4)
        assert_equal(np.maximum.reduce([4], initial=5), 5)
        assert_equal(np.maximum.reduce([5], initial=4), 5)
        assert_equal(np.minimum.reduce([4], initial=5), 4)

        # Check initial=None raises ValueError for both types of ufunc reductions
        assert_raises(ValueError, np.minimum.reduce, [], initial=None)
        assert_raises(ValueError, np.add.reduce, [], initial=None)

        # Check that np._NoValue gives default behavior.
        assert_equal(np.add.reduce([], initial=np._NoValue), 0)

        # Check that initial kwarg behaves as intended for dtype=object
        a = np.array([10], dtype=object)
        res = np.add.reduce(a, initial=5)
        assert_equal(res, 15)

    @pytest.mark.parametrize('axis', (0, 1, None))
    @pytest.mark.parametrize('where', (np.array([False, True, True]),
                                       np.array([[True], [False], [True]]),
                                       np.array([[True, False, False],
                                                 [False, True, False],
                                                 [False, True, True]])))
    def test_reduction_with_where(self, axis, where):
        a = np.arange(9.).reshape(3, 3)
        a_copy = a.copy()
        a_check = np.zeros_like(a)
        np.positive(a, out=a_check, where=where)

        res = np.add.reduce(a, axis=axis, where=where)
        check = a_check.sum(axis)
        assert_equal(res, check)
        # Check we do not overwrite elements of a internally.
        assert_array_equal(a, a_copy)

    @pytest.mark.parametrize(('axis', 'where'),
                             ((0, np.array([True, False, True])),
                              (1, [True, True, False]),
                              (None, True)))
    @pytest.mark.parametrize('initial', (-np.inf, 5.))
    def test_reduction_with_where_and_initial(self, axis, where, initial):
        a = np.arange(9.).reshape(3, 3)
        a_copy = a.copy()
        a_check = np.full(a.shape, -np.inf)
        np.positive(a, out=a_check, where=where)

        res = np.maximum.reduce(a, axis=axis, where=where, initial=initial)
        check = a_check.max(axis, initial=initial)
        assert_equal(res, check)

    def test_reduction_where_initial_needed(self):
        a = np.arange(9.).reshape(3, 3)
        m = [False, True, False]
        assert_raises(ValueError, np.maximum.reduce, a, where=m)

    def test_identityless_reduction_nonreorderable(self):
        a = np.array([[8.0, 2.0, 2.0], [1.0, 0.5, 0.25]])

        res = np.divide.reduce(a, axis=0)
        assert_equal(res, [8.0, 4.0, 8.0])

        res = np.divide.reduce(a, axis=1)
        assert_equal(res, [2.0, 8.0])

        res = np.divide.reduce(a, axis=())
        assert_equal(res, a)

        assert_raises(ValueError, np.divide.reduce, a, axis=(0, 1))

    def test_reduce_zero_axis(self):
        # If we have a n x m array and do a reduction with axis=1, then we are
        # doing n reductions, and each reduction takes an m-element array. For
        # a reduction operation without an identity, then:
        #   n > 0, m > 0: fine
        #   n = 0, m > 0: fine, doing 0 reductions of m-element arrays
        #   n > 0, m = 0: can't reduce a 0-element array, ValueError
        #   n = 0, m = 0: can't reduce a 0-element array, ValueError (for
        #     consistency with the above case)
        # This test doesn't actually look at return values, it just checks to
        # make sure that error we get an error in exactly those cases where we
        # expect one, and assumes the calculations themselves are done
        # correctly.

        def ok(f, *args, **kwargs):
            f(*args, **kwargs)

        def err(f, *args, **kwargs):
            assert_raises(ValueError, f, *args, **kwargs)

        def t(expect, func, n, m):
            expect(func, np.zeros((n, m)), axis=1)
            expect(func, np.zeros((m, n)), axis=0)
            expect(func, np.zeros((n // 2, n // 2, m)), axis=2)
            expect(func, np.zeros((n // 2, m, n // 2)), axis=1)
            expect(func, np.zeros((n, m // 2, m // 2)), axis=(1, 2))
            expect(func, np.zeros((m // 2, n, m // 2)), axis=(0, 2))
            expect(func, np.zeros((m // 3, m // 3, m // 3,
                                  n // 2, n // 2)),
                                 axis=(0, 1, 2))
            # Check what happens if the inner (resp. outer) dimensions are a
            # mix of zero and non-zero:
            expect(func, np.zeros((10, m, n)), axis=(0, 1))
            expect(func, np.zeros((10, n, m)), axis=(0, 2))
            expect(func, np.zeros((m, 10, n)), axis=0)
            expect(func, np.zeros((10, m, n)), axis=1)
            expect(func, np.zeros((10, n, m)), axis=2)

        # np.maximum is just an arbitrary ufunc with no reduction identity
        assert_equal(np.maximum.identity, None)
        t(ok, np.maximum.reduce, 30, 30)
        t(ok, np.maximum.reduce, 0, 30)
        t(err, np.maximum.reduce, 30, 0)
        t(err, np.maximum.reduce, 0, 0)
        err(np.maximum.reduce, [])
        np.maximum.reduce(np.zeros((0, 0)), axis=())

        # all of the combinations are fine for a reduction that has an
        # identity
        t(ok, np.add.reduce, 30, 30)
        t(ok, np.add.reduce, 0, 30)
        t(ok, np.add.reduce, 30, 0)
        t(ok, np.add.reduce, 0, 0)
        np.add.reduce([])
        np.add.reduce(np.zeros((0, 0)), axis=())

        # OTOH, accumulate always makes sense for any combination of n and m,
        # because it maps an m-element array to an m-element array. These
        # tests are simpler because accumulate doesn't accept multiple axes.
        for uf in (np.maximum, np.add):
            uf.accumulate(np.zeros((30, 0)), axis=0)
            uf.accumulate(np.zeros((0, 30)), axis=0)
            uf.accumulate(np.zeros((30, 30)), axis=0)
            uf.accumulate(np.zeros((0, 0)), axis=0)

    def test_safe_casting(self):
        # In old versions of numpy, in-place operations used the 'unsafe'
        # casting rules. In versions >= 1.10, 'same_kind' is the
        # default and an exception is raised instead of a warning.
        # when 'same_kind' is not satisfied.
        a = np.array([1, 2, 3], dtype=int)
        # Non-in-place addition is fine
        assert_array_equal(assert_no_warnings(np.add, a, 1.1),
                           [2.1, 3.1, 4.1])
        assert_raises(TypeError, np.add, a, 1.1, out=a)

        def add_inplace(a, b):
            a += b

        assert_raises(TypeError, add_inplace, a, 1.1)
        # Make sure that explicitly overriding the exception is allowed:
        assert_no_warnings(np.add, a, 1.1, out=a, casting="unsafe")
        assert_array_equal(a, [2, 3, 4])

    def test_ufunc_custom_out(self):
        # Test ufunc with built in input types and custom output type

        a = np.array([0, 1, 2], dtype='i8')
        b = np.array([0, 1, 2], dtype='i8')
        c = np.empty(3, dtype=_rational_tests.rational)

        # Output must be specified so numpy knows what
        # ufunc signature to look for
        result = _rational_tests.test_add(a, b, c)
        target = np.array([0, 2, 4], dtype=_rational_tests.rational)
        assert_equal(result, target)

        # The new resolution means that we can (usually) find custom loops
        # as long as they match exactly:
        result = _rational_tests.test_add(a, b)
        assert_equal(result, target)

        # This works even more generally, so long the default common-dtype
        # promoter works out:
        result = _rational_tests.test_add(a, b.astype(np.uint16), out=c)
        assert_equal(result, target)

        # But, it can be fooled, e.g. (use scalars, which forces legacy
        # type resolution to kick in, which then fails):
        with assert_raises(TypeError):
            _rational_tests.test_add(a, np.uint16(2))

    def test_operand_flags(self):
        a = np.arange(16, dtype='l').reshape(4, 4)
        b = np.arange(9, dtype='l').reshape(3, 3)
        opflag_tests.inplace_add(a[:-1, :-1], b)
        assert_equal(a, np.array([[0, 2, 4, 3], [7, 9, 11, 7],
            [14, 16, 18, 11], [12, 13, 14, 15]], dtype='l'))

        a = np.array(0)
        opflag_tests.inplace_add(a, 3)
        assert_equal(a, 3)
        opflag_tests.inplace_add(a, [3, 4])
        assert_equal(a, 10)

    def test_struct_ufunc(self):
        import numpy.core._struct_ufunc_tests as struct_ufunc

        a = np.array([(1, 2, 3)], dtype='u8,u8,u8')
        b = np.array([(1, 2, 3)], dtype='u8,u8,u8')

        result = struct_ufunc.add_triplet(a, b)
        assert_equal(result, np.array([(2, 4, 6)], dtype='u8,u8,u8'))
        assert_raises(RuntimeError, struct_ufunc.register_fail)

    def test_custom_ufunc(self):
        a = np.array(
            [_rational_tests.rational(1, 2),
             _rational_tests.rational(1, 3),
             _rational_tests.rational(1, 4)],
            dtype=_rational_tests.rational)
        b = np.array(
            [_rational_tests.rational(1, 2),
             _rational_tests.rational(1, 3),
             _rational_tests.rational(1, 4)],
            dtype=_rational_tests.rational)

        result = _rational_tests.test_add_rationals(a, b)
        expected = np.array(
            [_rational_tests.rational(1),
             _rational_tests.rational(2, 3),
             _rational_tests.rational(1, 2)],
            dtype=_rational_tests.rational)
        assert_equal(result, expected)

    def test_custom_ufunc_forced_sig(self):
        # gh-9351 - looking for a non-first userloop would previously hang
        with assert_raises(TypeError):
            np.multiply(_rational_tests.rational(1), 1,
                        signature=(_rational_tests.rational, int, None))

    def test_custom_array_like(self):

        class MyThing:
            __array_priority__ = 1000

            rmul_count = 0
            getitem_count = 0

            def __init__(self, shape):
                self.shape = shape

            def __len__(self):
                return self.shape[0]

            def __getitem__(self, i):
                MyThing.getitem_count += 1
                if not isinstance(i, tuple):
                    i = (i,)
                if len(i) > self.ndim:
                    raise IndexError("boo")

                return MyThing(self.shape[len(i):])

            def __rmul__(self, other):
                MyThing.rmul_count += 1
                return self

        np.float64(5)*MyThing((3, 3))
        assert_(MyThing.rmul_count == 1, MyThing.rmul_count)
        assert_(MyThing.getitem_count <= 2, MyThing.getitem_count)

    def test_inplace_fancy_indexing(self):

        a = np.arange(10)
        np.add.at(a, [2, 5, 2], 1)
        assert_equal(a, [0, 1, 4, 3, 4, 6, 6, 7, 8, 9])

        a = np.arange(10)
        b = np.array([100, 100, 100])
        np.add.at(a, [2, 5, 2], b)
        assert_equal(a, [0, 1, 202, 3, 4, 105, 6, 7, 8, 9])

        a = np.arange(9).reshape(3, 3)
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        np.add.at(a, (slice(None), [1, 2, 1]), b)
        assert_equal(a, [[0, 201, 102], [3, 404, 205], [6, 607, 308]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), slice(None), [1, 2, 1]), b)
        assert_equal(a,
            [[[0, 401, 202],
              [3, 404, 205],
              [6, 407, 208]],

             [[9, 410, 211],
              [12, 413, 214],
              [15, 416, 217]],

             [[18, 419, 220],
              [21, 422, 223],
              [24, 425, 226]]])

        a = np.arange(9).reshape(3, 3)
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        np.add.at(a, ([1, 2, 1], slice(None)), b)
        assert_equal(a, [[0, 1, 2], [403, 404, 405], [206, 207, 208]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), [1, 2, 1], slice(None)), b)
        assert_equal(a,
            [[[0,  1,  2],
              [203, 404, 605],
              [106, 207, 308]],

             [[9,  10, 11],
              [212, 413, 614],
              [115, 216, 317]],

             [[18, 19, 20],
              [221, 422, 623],
              [124, 225, 326]]])

        a = np.arange(9).reshape(3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (0, [1, 2, 1]), b)
        assert_equal(a, [[0, 401, 202], [3, 4, 5], [6, 7, 8]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, ([1, 2, 1], 0, slice(None)), b)
        assert_equal(a,
            [[[0,  1,  2],
              [3,  4,  5],
              [6,  7,  8]],

             [[209, 410, 611],
              [12,  13, 14],
              [15,  16, 17]],

             [[118, 219, 320],
              [21,  22, 23],
              [24,  25, 26]]])

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        np.add.at(a, (slice(None), slice(None), slice(None)), b)
        assert_equal(a,
            [[[100, 201, 302],
              [103, 204, 305],
              [106, 207, 308]],

             [[109, 210, 311],
              [112, 213, 314],
              [115, 216, 317]],

             [[118, 219, 320],
              [121, 222, 323],
              [124, 225, 326]]])

        a = np.arange(10)
        np.negative.at(a, [2, 5, 2])
        assert_equal(a, [0, 1, 2, 3, 4, -5, 6, 7, 8, 9])

        # Test 0-dim array
        a = np.array(0)
        np.add.at(a, (), 1)
        assert_equal(a, 1)

        assert_raises(IndexError, np.add.at, a, 0, 1)
        assert_raises(IndexError, np.add.at, a, [], 1)

        # Test mixed dtypes
        a = np.arange(10)
        np.power.at(a, [1, 2, 3, 2], 3.5)
        assert_equal(a, np.array([0, 1, 4414, 46, 4, 5, 6, 7, 8, 9]))

        # Test boolean indexing and boolean ufuncs
        a = np.arange(10)
        index = a % 2 == 0
        np.equal.at(a, index, [0, 2, 4, 6, 8])
        assert_equal(a, [1, 1, 1, 3, 1, 5, 1, 7, 1, 9])

        # Test unary operator
        a = np.arange(10, dtype='u4')
        np.invert.at(a, [2, 5, 2])
        assert_equal(a, [0, 1, 2, 3, 4, 5 ^ 0xffffffff, 6, 7, 8, 9])

        # Test empty subspace
        orig = np.arange(4)
        a = orig[:, None][:, 0:0]
        np.add.at(a, [0, 1], 3)
        assert_array_equal(orig, np.arange(4))

        # Test with swapped byte order
        index = np.array([1, 2, 1], np.dtype('i').newbyteorder())
        values = np.array([1, 2, 3, 4], np.dtype('f').newbyteorder())
        np.add.at(values, index, 3)
        assert_array_equal(values, [1, 8, 6, 4])

        # Test exception thrown
        values = np.array(['a', 1], dtype=object)
        assert_raises(TypeError, np.add.at, values, [0, 1], 1)
        assert_array_equal(values, np.array(['a', 1], dtype=object))

        # Test multiple output ufuncs raise error, gh-5665
        assert_raises(ValueError, np.modf.at, np.arange(10), [1])

    def test_reduce_arguments(self):
        f = np.add.reduce
        d = np.ones((5,2), dtype=int)
        o = np.ones((2,), dtype=d.dtype)
        r = o * 5
        assert_equal(f(d), r)
        # a, axis=0, dtype=None, out=None, keepdims=False
        assert_equal(f(d, axis=0), r)
        assert_equal(f(d, 0), r)
        assert_equal(f(d, 0, dtype=None), r)
        assert_equal(f(d, 0, dtype='i'), r)
        assert_equal(f(d, 0, 'i'), r)
        assert_equal(f(d, 0, None), r)
        assert_equal(f(d, 0, None, out=None), r)
        assert_equal(f(d, 0, None, out=o), r)
        assert_equal(f(d, 0, None, o), r)
        assert_equal(f(d, 0, None, None), r)
        assert_equal(f(d, 0, None, None, keepdims=False), r)
        assert_equal(f(d, 0, None, None, True), r.reshape((1,) + r.shape))
        assert_equal(f(d, 0, None, None, False, 0), r)
        assert_equal(f(d, 0, None, None, False, initial=0), r)
        assert_equal(f(d, 0, None, None, False, 0, True), r)
        assert_equal(f(d, 0, None, None, False, 0, where=True), r)
        # multiple keywords
        assert_equal(f(d, axis=0, dtype=None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, dtype=None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, None, out=None, keepdims=False), r)
        assert_equal(f(d, 0, None, out=None, keepdims=False, initial=0,
                       where=True), r)

        # too little
        assert_raises(TypeError, f)
        # too much
        assert_raises(TypeError, f, d, 0, None, None, False, 0, True, 1)
        # invalid axis
        assert_raises(TypeError, f, d, "invalid")
        assert_raises(TypeError, f, d, axis="invalid")
        assert_raises(TypeError, f, d, axis="invalid", dtype=None,
                      keepdims=True)
        # invalid dtype
        assert_raises(TypeError, f, d, 0, "invalid")
        assert_raises(TypeError, f, d, dtype="invalid")
        assert_raises(TypeError, f, d, dtype="invalid", out=None)
        # invalid out
        assert_raises(TypeError, f, d, 0, None, "invalid")
        assert_raises(TypeError, f, d, out="invalid")
        assert_raises(TypeError, f, d, out="invalid", dtype=None)
        # keepdims boolean, no invalid value
        # assert_raises(TypeError, f, d, 0, None, None, "invalid")
        # assert_raises(TypeError, f, d, keepdims="invalid", axis=0, dtype=None)
        # invalid mix
        assert_raises(TypeError, f, d, 0, keepdims="invalid", dtype="invalid",
                     out=None)

        # invalid keyword
        assert_raises(TypeError, f, d, axis=0, dtype=None, invalid=0)
        assert_raises(TypeError, f, d, invalid=0)
        assert_raises(TypeError, f, d, 0, keepdims=True, invalid="invalid",
                      out=None)
        assert_raises(TypeError, f, d, axis=0, dtype=None, keepdims=True,
                      out=None, invalid=0)
        assert_raises(TypeError, f, d, axis=0, dtype=None,
                      out=None, invalid=0)

    def test_structured_equal(self):
        # https://github.com/numpy/numpy/issues/4855

        class MyA(np.ndarray):
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return getattr(ufunc, method)(*(input.view(np.ndarray)
                                              for input in inputs), **kwargs)
        a = np.arange(12.).reshape(4,3)
        ra = a.view(dtype=('f8,f8,f8')).squeeze()
        mra = ra.view(MyA)

        target = np.array([ True, False, False, False], dtype=bool)
        assert_equal(np.all(target == (mra == ra[0])), True)

    def test_scalar_equal(self):
        # Scalar comparisons should always work, without deprecation warnings.
        # even when the ufunc fails.
        a = np.array(0.)
        b = np.array('a')
        assert_(a != b)
        assert_(b != a)
        assert_(not (a == b))
        assert_(not (b == a))

    def test_NotImplemented_not_returned(self):
        # See gh-5964 and gh-2091. Some of these functions are not operator
        # related and were fixed for other reasons in the past.
        binary_funcs = [
            np.power, np.add, np.subtract, np.multiply, np.divide,
            np.true_divide, np.floor_divide, np.bitwise_and, np.bitwise_or,
            np.bitwise_xor, np.left_shift, np.right_shift, np.fmax,
            np.fmin, np.fmod, np.hypot, np.logaddexp, np.logaddexp2,
            np.maximum, np.minimum, np.mod,
            np.greater, np.greater_equal, np.less, np.less_equal,
            np.equal, np.not_equal]

        a = np.array('1')
        b = 1
        c = np.array([1., 2.])
        for f in binary_funcs:
            assert_raises(TypeError, f, a, b)
            assert_raises(TypeError, f, c, a)

    @pytest.mark.parametrize("ufunc",
             [np.logical_and, np.logical_or])  # logical_xor object loop is bad
    @pytest.mark.parametrize("signature",
             [(None, None, object), (object, None, None),
              (None, object, None)])
    def test_logical_ufuncs_object_signatures(self, ufunc, signature):
        a = np.array([True, None, False], dtype=object)
        res = ufunc(a, a, signature=signature)
        assert res.dtype == object

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    @pytest.mark.parametrize("signature",
                 [(bool, None, object), (object, None, bool),
                  (None, object, bool)])
    def test_logical_ufuncs_mixed_object_signatures(self, ufunc, signature):
        # Most mixed signatures fail (except those with bool out, e.g. `OO->?`)
        a = np.array([True, None, False])
        with pytest.raises(TypeError):
            ufunc(a, a, signature=signature)

    @pytest.mark.parametrize("ufunc",
            [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_support_anything(self, ufunc):
        # The logical ufuncs support even input that can't be promoted:
        a = np.array('1')
        c = np.array([1., 2.])
        assert_array_equal(ufunc(a, c), ufunc([True, True], True))
        assert ufunc.reduce(a) == True
        # check that the output has no effect:
        out = np.zeros(2, dtype=np.int32)
        expected = ufunc([True, True], True).astype(out.dtype)
        assert_array_equal(ufunc(a, c, out=out), expected)
        out = np.zeros((), dtype=np.int32)
        assert ufunc.reduce(a, out=out) == True
        # Last check, test reduction when out and a match (the complexity here
        # is that the "i,i->?" may seem right, but should not match.
        a = np.array([3], dtype="i")
        out = np.zeros((), dtype=a.dtype)
        assert ufunc.reduce(a, out=out) == 1

    @pytest.mark.parametrize("ufunc",
             [np.logical_and, np.logical_or, np.logical_xor])
    def test_logical_ufuncs_out_cast_check(self, ufunc):
        a = np.array('1')
        c = np.array([1., 2.])
        out = a.copy()
        with pytest.raises(TypeError):
            # It would be safe, but not equiv casting:
            ufunc(a, c, out=out, casting="equiv")

    def test_reducelike_byteorder_resolution(self):
        # See gh-20699, byte-order changes need some extra care in the type
        # resolution to make the following succeed:
        arr_be = np.arange(10, dtype=">i8")
        arr_le = np.arange(10, dtype="<i8")

        assert np.add.reduce(arr_be) == np.add.reduce(arr_le)
        assert_array_equal(
            np.add.accumulate(arr_be), np.add.accumulate(arr_le))
        assert_array_equal(
            np.add.reduceat(arr_be, [1]), np.add.reduceat(arr_le, [1]))

    def test_reducelike_out_promotes(self):
        # Check that the out argument to reductions is considered for
        # promotion.  See also gh-20455.
        # Note that these paths could prefer `initial=` in the future and
        # do not up-cast to the default integer for add and prod
        arr = np.ones(1000, dtype=np.uint8)
        out = np.zeros((), dtype=np.uint16)
        assert np.add.reduce(arr, out=out) == 1000
        arr[:10] = 2
        assert np.multiply.reduce(arr, out=out) == 2**10

        # For legacy dtypes, the signature currently has to be forced if `out=`
        # is passed.  The two paths below should differ, without `dtype=` the
        # expected result should be: `np.prod(arr.astype("f8")).astype("f4")`!
        arr = np.full(5, 2**25-1, dtype=np.int64)

        # float32 and int64 promote to float64:
        res = np.zeros((), dtype=np.float32)
        # If `dtype=` is passed, the calculation is forced to float32:
        single_res = np.zeros((), dtype=np.float32)
        np.multiply.reduce(arr, out=single_res, dtype=np.float32)
        assert single_res != res

    def test_reducelike_output_needs_identical_cast(self):
        # Checks the case where the we have a simple byte-swap works, maily
        # tests that this is not rejected directly.
        # (interesting because we require descriptor identity in reducelikes).
        arr = np.ones(20, dtype="f8")
        out = np.empty((), dtype=arr.dtype.newbyteorder())
        expected = np.add.reduce(arr)
        np.add.reduce(arr, out=out)
        assert_array_equal(expected, out)
        # Check reduceat:
        out = np.empty(2, dtype=arr.dtype.newbyteorder())
        expected = np.add.reduceat(arr, [0, 1])
        np.add.reduceat(arr, [0, 1], out=out)
        assert_array_equal(expected, out)
        # And accumulate:
        out = np.empty(arr.shape, dtype=arr.dtype.newbyteorder())
        expected = np.add.accumulate(arr)
        np.add.accumulate(arr, out=out)
        assert_array_equal(expected, out)

    def test_reduce_noncontig_output(self):
        # Check that reduction deals with non-contiguous output arrays
        # appropriately.
        #
        # gh-8036

        x = np.arange(7*13*8, dtype=np.int16).reshape(7, 13, 8)
        x = x[4:6,1:11:6,1:5].transpose(1, 2, 0)
        y_base = np.arange(4*4, dtype=np.int16).reshape(4, 4)
        y = y_base[::2,:]

        y_base_copy = y_base.copy()

        r0 = np.add.reduce(x, out=y.copy(), axis=2)
        r1 = np.add.reduce(x, out=y, axis=2)

        # The results should match, and y_base shouldn't get clobbered
        assert_equal(r0, r1)
        assert_equal(y_base[1,:], y_base_copy[1,:])
        assert_equal(y_base[3,:], y_base_copy[3,:])

    @pytest.mark.parametrize("with_cast", [True, False])
    def test_reduceat_and_accumulate_out_shape_mismatch(self, with_cast):
        # Should raise an error mentioning "shape" or "size"
        arr = np.arange(5)
        out = np.arange(3)  # definitely wrong shape
        if with_cast:
            # If a cast is necessary on the output, we can be sure to use
            # the generic NpyIter (non-fast) path.
            out = out.astype(np.float64)

        with pytest.raises(ValueError, match="(shape|size)"):
            np.add.reduceat(arr, [0, 3], out=out)

        with pytest.raises(ValueError, match="(shape|size)"):
            np.add.accumulate(arr, out=out)

    @pytest.mark.parametrize('out_shape',
                             [(), (1,), (3,), (1, 1), (1, 3), (4, 3)])
    @pytest.mark.parametrize('keepdims', [True, False])
    @pytest.mark.parametrize('f_reduce', [np.add.reduce, np.minimum.reduce])
    def test_reduce_wrong_dimension_output(self, f_reduce, keepdims, out_shape):
        # Test that we're not incorrectly broadcasting dimensions.
        # See gh-15144 (failed for np.add.reduce previously).
        a = np.arange(12.).reshape(4, 3)
        out = np.empty(out_shape, a.dtype)

        correct_out = f_reduce(a, axis=0, keepdims=keepdims)
        if out_shape != correct_out.shape:
            with assert_raises(ValueError):
                f_reduce(a, axis=0, out=out, keepdims=keepdims)
        else:
            check = f_reduce(a, axis=0, out=out, keepdims=keepdims)
            assert_(check is out)
            assert_array_equal(check, correct_out)

    def test_reduce_output_does_not_broadcast_input(self):
        # Test that the output shape cannot broadcast an input dimension
        # (it never can add dimensions, but it might expand an existing one)
        a = np.ones((1, 10))
        out_correct = (np.empty((1, 1)))
        out_incorrect = np.empty((3, 1))
        np.add.reduce(a, axis=-1, out=out_correct, keepdims=True)
        np.add.reduce(a, axis=-1, out=out_correct[:, 0], keepdims=False)
        with assert_raises(ValueError):
            np.add.reduce(a, axis=-1, out=out_incorrect, keepdims=True)
        with assert_raises(ValueError):
            np.add.reduce(a, axis=-1, out=out_incorrect[:, 0], keepdims=False)

    def test_reduce_output_subclass_ok(self):
        class MyArr(np.ndarray):
            pass

        out = np.empty(())
        np.add.reduce(np.ones(5), out=out)  # no subclass, all fine
        out = out.view(MyArr)
        assert np.add.reduce(np.ones(5), out=out) is out
        assert type(np.add.reduce(out)) is MyArr

    def test_no_doc_string(self):
        # gh-9337
        assert_('\n' not in umt.inner1d_no_doc.__doc__)

    def test_invalid_args(self):
        # gh-7961
        exc = pytest.raises(TypeError, np.sqrt, None)
        # minimally check the exception text
        assert exc.match('loop of ufunc does not support')

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_not_finite(self, nat):
        try:
            assert not np.isfinite(nat)
        except TypeError:
            pass  # ok, just not implemented

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_nan(self, nat):
        try:
            assert np.isnan(nat)
        except TypeError:
            pass  # ok, just not implemented

    @pytest.mark.parametrize('nat', [np.datetime64('nat'), np.timedelta64('nat')])
    def test_nat_is_not_inf(self, nat):
        try:
            assert not np.isinf(nat)
        except TypeError:
            pass  # ok, just not implemented


@pytest.mark.parametrize('ufunc', [getattr(np, x) for x in dir(np)
                                if isinstance(getattr(np, x), np.ufunc)])
def test_ufunc_types(ufunc):
    '''
    Check all ufuncs that the correct type is returned. Avoid
    object and boolean types since many operations are not defined for
    for them.

    Choose the shape so even dot and matmul will succeed
    '''
    for typ in ufunc.types:
        # types is a list of strings like ii->i
        if 'O' in typ or '?' in typ:
            continue
        inp, out = typ.split('->')
        args = [np.ones((3, 3), t) for t in inp]
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            res = ufunc(*args)
        if isinstance(res, tuple):
            outs = tuple(out)
            assert len(res) == len(outs)
            for r, t in zip(res, outs):
                assert r.dtype == np.dtype(t)
        else:
            assert res.dtype == np.dtype(out)

@pytest.mark.parametrize('ufunc', [getattr(np, x) for x in dir(np)
                                if isinstance(getattr(np, x), np.ufunc)])
def test_ufunc_noncontiguous(ufunc):
    '''
    Check that contiguous and non-contiguous calls to ufuncs
    have the same results for values in range(9)
    '''
    for typ in ufunc.types:
        # types is a list of strings like ii->i
        if any(set('O?mM') & set(typ)):
            # bool, object, datetime are too irregular for this simple test
            continue
        inp, out = typ.split('->')
        args_c = [np.empty(6, t) for t in inp]
        args_n = [np.empty(18, t)[::3] for t in inp]
        for a in args_c:
            a.flat = range(1,7)
        for a in args_n:
            a.flat = range(1,7)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            res_c = ufunc(*args_c)
            res_n = ufunc(*args_n)
        if len(out) == 1:
            res_c = (res_c,)
            res_n = (res_n,)
        for c_ar, n_ar in zip(res_c, res_n):
            dt = c_ar.dtype
            if np.issubdtype(dt, np.floating):
                # for floating point results allow a small fuss in comparisons
                # since different algorithms (libm vs. intrinsics) can be used
                # for different input strides
                res_eps = np.finfo(dt).eps
                tol = 2*res_eps
                assert_allclose(res_c, res_n, atol=tol, rtol=tol)
            else:
                assert_equal(c_ar, n_ar)


@pytest.mark.parametrize('ufunc', [np.sign, np.equal])
def test_ufunc_warn_with_nan(ufunc):
    # issue gh-15127
    # test that calling certain ufuncs with a non-standard `nan` value does not
    # emit a warning
    # `b` holds a 64 bit signaling nan: the most significant bit of the
    # significand is zero.
    b = np.array([0x7ff0000000000001], 'i8').view('f8')
    assert np.isnan(b)
    if ufunc.nin == 1:
        ufunc(b)
    elif ufunc.nin == 2:
        ufunc(b, b.copy())
    else:
        raise ValueError('ufunc with more than 2 inputs')


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_ufunc_casterrors():
    # Tests that casting errors are correctly reported and buffers are
    # cleared.
    # The following array can be added to itself as an object array, but
    # the result cannot be cast to an integer output:
    value = 123  # relies on python cache (leak-check will still find it)
    arr = np.array([value] * int(np.BUFSIZE * 1.5) +
                   ["string"] +
                   [value] * int(1.5 * np.BUFSIZE), dtype=object)
    out = np.ones(len(arr), dtype=np.intp)

    count = sys.getrefcount(value)
    with pytest.raises(ValueError):
        # Output casting failure:
        np.add(arr, arr, out=out, casting="unsafe")

    assert count == sys.getrefcount(value)
    # output is unchanged after the error, this shows that the iteration
    # was aborted (this is not necessarily defined behaviour)
    assert out[-1] == 1

    with pytest.raises(ValueError):
        # Input casting failure:
        np.add(arr, arr, out=out, dtype=np.intp, casting="unsafe")

    assert count == sys.getrefcount(value)
    # output is unchanged after the error, this shows that the iteration
    # was aborted (this is not necessarily defined behaviour)
    assert out[-1] == 1


def test_trivial_loop_invalid_cast():
    # This tests the fast-path "invalid cast", see gh-19904.
    with pytest.raises(TypeError,
            match="cast ufunc 'add' input 0"):
        # the void dtype definitely cannot cast to double:
        np.add(np.array(1, "i,i"), 3, signature="dd->d")


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
@pytest.mark.parametrize("offset",
        [0, np.BUFSIZE//2, int(1.5*np.BUFSIZE)])
def test_reduce_casterrors(offset):
    # Test reporting of casting errors in reductions, we test various
    # offsets to where the casting error will occur, since these may occur
    # at different places during the reduction procedure. For example
    # the first item may be special.
    value = 123  # relies on python cache (leak-check will still find it)
    arr = np.array([value] * offset +
                   ["string"] +
                   [value] * int(1.5 * np.BUFSIZE), dtype=object)
    out = np.array(-1, dtype=np.intp)

    count = sys.getrefcount(value)
    with pytest.raises(ValueError, match="invalid literal"):
        # This is an unsafe cast, but we currently always allow that.
        # Note that the double loop is picked, but the cast fails.
        np.add.reduce(arr, dtype=np.intp, out=out)
    assert count == sys.getrefcount(value)
    # If an error occurred during casting, the operation is done at most until
    # the error occurs (the result of which would be `value * offset`) and -1
    # if the error happened immediately.
    # This does not define behaviour, the output is invalid and thus undefined
    assert out[()] < value * offset


@pytest.mark.parametrize("method",
        [np.add.accumulate, np.add.reduce,
         pytest.param(lambda x: np.add.reduceat(x, [0]), id="reduceat"),
         pytest.param(lambda x: np.log.at(x, [2]), id="at")])
def test_ufunc_methods_floaterrors(method):
    # adding inf and -inf (or log(-inf) creates an invalid float and warns
    arr = np.array([np.inf, 0, -np.inf])
    with np.errstate(all="warn"):
        with pytest.warns(RuntimeWarning, match="invalid value"):
            method(arr)

    arr = np.array([np.inf, 0, -np.inf])
    with np.errstate(all="raise"):
        with pytest.raises(FloatingPointError):
            method(arr)import builtins
import os
import sys
import mmap
import ctypes as ct
import array as _array
import datetime as dt
import enum
from abc import abstractmethod
from types import TracebackType, MappingProxyType
from contextlib import ContextDecorator

if sys.version_info >= (3, 9):
    from types import GenericAlias

from numpy._pytesttester import PytestTester
from numpy.core._internal import _ctypes

from numpy.typing import (
    # Arrays
    ArrayLike,
    NDArray,
    _SupportsArray,
    _NestedSequence,
    _FiniteNestedSequence,
    _SupportsArray,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeNumber_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
    _ArrayLikeObject_co,
    _ArrayLikeStr_co,
    _ArrayLikeBytes_co,

    # DTypes
    DTypeLike,
    _SupportsDType,
    _VoidDTypeLike,

    # Shapes
    _Shape,
    _ShapeLike,

    # Scalars
    _CharLike_co,
    _BoolLike_co,
    _IntLike_co,
    _FloatLike_co,
    _ComplexLike_co,
    _TD64Like_co,
    _NumberLike_co,
    _ScalarLike_co,

    # `number` precision
    NBitBase,
    _256Bit,
    _128Bit,
    _96Bit,
    _80Bit,
    _64Bit,
    _32Bit,
    _16Bit,
    _8Bit,
    _NBitByte,
    _NBitShort,
    _NBitIntC,
    _NBitIntP,
    _NBitInt,
    _NBitLongLong,
    _NBitHalf,
    _NBitSingle,
    _NBitDouble,
    _NBitLongDouble,

    # Character codes
    _BoolCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _Complex64Codes,
    _Complex128Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _IntPCodes,
    _IntCodes,
    _LongLongCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntPCodes,
    _UIntCodes,
    _ULongLongCodes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
    _DT64Codes,
    _TD64Codes,
    _StrCodes,
    _BytesCodes,
    _VoidCodes,
    _ObjectCodes,

    # Ufuncs
    _UFunc_Nin1_Nout1,
    _UFunc_Nin2_Nout1,
    _UFunc_Nin1_Nout2,
    _UFunc_Nin2_Nout2,
    _GUFunc_Nin2_Nout1,
)

from numpy.typing._callable import (
    _BoolOp,
    _BoolBitOp,
    _BoolSub,
    _BoolTrueDiv,
    _BoolMod,
    _BoolDivMod,
    _TD64Div,
    _IntTrueDiv,
    _UnsignedIntOp,
    _UnsignedIntBitOp,
    _UnsignedIntMod,
    _UnsignedIntDivMod,
    _SignedIntOp,
    _SignedIntBitOp,
    _SignedIntMod,
    _SignedIntDivMod,
    _FloatOp,
    _FloatMod,
    _FloatDivMod,
    _ComplexOp,
    _NumberOp,
    _ComparisonOp,
)

# NOTE: Numpy's mypy plugin is used for removing the types unavailable
# to the specific platform
from numpy.typing._extended_precision import (
    uint128 as uint128,
    uint256 as uint256,
    int128 as int128,
    int256 as int256,
    float80 as float80,
    float96 as float96,
    float128 as float128,
    float256 as float256,
    complex160 as complex160,
    complex192 as complex192,
    complex256 as complex256,
    complex512 as complex512,
)

from typing import (
    Literal as L,
    Any,
    ByteString,
    Callable,
    Container,
    Callable,
    Dict,
    Generic,
    IO,
    Iterable,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    overload,
    Sequence,
    Sized,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    Text,
    Tuple,
    Type,
    TypeVar,
    Union,
    Protocol,
    SupportsIndex,
    Final,
    final,
    ClassVar,
    Set,
)

# Ensures that the stubs are picked up
from numpy import (
    ctypeslib as ctypeslib,
    fft as fft,
    lib as lib,
    linalg as linalg,
    ma as ma,
    matrixlib as matrixlib,
    polynomial as polynomial,
    random as random,
    testing as testing,
    version as version,
)

from numpy.core import defchararray, records
char = defchararray
rec = records

from numpy.core.function_base import (
    linspace as linspace,
    logspace as logspace,
    geomspace as geomspace,
)

from numpy.core.fromnumeric import (
    take as take,
    reshape as reshape,
    choose as choose,
    repeat as repeat,
    put as put,
    swapaxes as swapaxes,
    transpose as transpose,
    partition as partition,
    argpartition as argpartition,
    sort as sort,
    argsort as argsort,
    argmax as argmax,
    argmin as argmin,
    searchsorted as searchsorted,
    resize as resize,
    squeeze as squeeze,
    diagonal as diagonal,
    trace as trace,
    ravel as ravel,
    nonzero as nonzero,
    shape as shape,
    compress as compress,
    clip as clip,
    sum as sum,
    all as all,
    any as any,
    cumsum as cumsum,
    ptp as ptp,
    amax as amax,
    amin as amin,
    prod as prod,
    cumprod as cumprod,
    ndim as ndim,
    size as size,
    around as around,
    mean as mean,
    std as std,
    var as var,
)

from numpy.core._asarray import (
    require as require,
)

from numpy.core._type_aliases import (
    sctypes as sctypes,
    sctypeDict as sctypeDict,
)

from numpy.core._ufunc_config import (
    seterr as seterr,
    geterr as geterr,
    setbufsize as setbufsize,
    getbufsize as getbufsize,
    seterrcall as seterrcall,
    geterrcall as geterrcall,
    _ErrKind,
    _ErrFunc,
    _ErrDictOptional,
)

from numpy.core.arrayprint import (
    set_printoptions as set_printoptions,
    get_printoptions as get_printoptions,
    array2string as array2string,
    format_float_scientific as format_float_scientific,
    format_float_positional as format_float_positional,
    array_repr as array_repr,
    array_str as array_str,
    set_string_function as set_string_function,
    printoptions as printoptions,
)

from numpy.core.einsumfunc import (
    einsum as einsum,
    einsum_path as einsum_path,
)

from numpy.core.multiarray import (
    ALLOW_THREADS as ALLOW_THREADS,
    BUFSIZE as BUFSIZE,
    CLIP as CLIP,
    MAXDIMS as MAXDIMS,
    MAY_SHARE_BOUNDS as MAY_SHARE_BOUNDS,
    MAY_SHARE_EXACT as MAY_SHARE_EXACT,
    RAISE as RAISE,
    WRAP as WRAP,
    tracemalloc_domain as tracemalloc_domain,
    array as array,
    empty_like as empty_like,
    empty as empty,
    zeros as zeros,
    concatenate as concatenate,
    inner as inner,
    where as where,
    lexsort as lexsort,
    can_cast as can_cast,
    min_scalar_type as min_scalar_type,
    result_type as result_type,
    dot as dot,
    vdot as vdot,
    bincount as bincount,
    copyto as copyto,
    putmask as putmask,
    packbits as packbits,
    unpackbits as unpackbits,
    shares_memory as shares_memory,
    may_share_memory as may_share_memory,
    asarray as asarray,
    asanyarray as asanyarray,
    ascontiguousarray as ascontiguousarray,
    asfortranarray as asfortranarray,
    arange as arange,
    busday_count as busday_count,
    busday_offset as busday_offset,
    compare_chararrays as compare_chararrays,
    datetime_as_string as datetime_as_string,
    datetime_data as datetime_data,
    frombuffer as frombuffer,
    fromfile as fromfile,
    fromiter as fromiter,
    is_busday as is_busday,
    promote_types as promote_types,
    seterrobj as seterrobj,
    geterrobj as geterrobj,
    fromstring as fromstring,
    frompyfunc as frompyfunc,
    nested_iters as nested_iters,
    flagsobj,
)

from numpy.core.numeric import (
    zeros_like as zeros_like,
    ones as ones,
    ones_like as ones_like,
    full as full,
    full_like as full_like,
    count_nonzero as count_nonzero,
    isfortran as isfortran,
    argwhere as argwhere,
    flatnonzero as flatnonzero,
    correlate as correlate,
    convolve as convolve,
    outer as outer,
    tensordot as tensordot,
    roll as roll,
    rollaxis as rollaxis,
    moveaxis as moveaxis,
    cross as cross,
    indices as indices,
    fromfunction as fromfunction,
    isscalar as isscalar,
    binary_repr as binary_repr,
    base_repr as base_repr,
    identity as identity,
    allclose as allclose,
    isclose as isclose,
    array_equal as array_equal,
    array_equiv as array_equiv,
)

from numpy.core.numerictypes import (
    maximum_sctype as maximum_sctype,
    issctype as issctype,
    obj2sctype as obj2sctype,
    issubclass_ as issubclass_,
    issubsctype as issubsctype,
    issubdtype as issubdtype,
    sctype2char as sctype2char,
    find_common_type as find_common_type,
    nbytes as nbytes,
    cast as cast,
    ScalarType as ScalarType,
    typecodes as typecodes,
)

from numpy.core.shape_base import (
    atleast_1d as atleast_1d,
    atleast_2d as atleast_2d,
    atleast_3d as atleast_3d,
    block as block,
    hstack as hstack,
    stack as stack,
    vstack as vstack,
)

from numpy.lib import (
    emath as emath,
)

from numpy.lib.arraypad import (
    pad as pad,
)

from numpy.lib.arraysetops import (
    ediff1d as ediff1d,
    intersect1d as intersect1d,
    setxor1d as setxor1d,
    union1d as union1d,
    setdiff1d as setdiff1d,
    unique as unique,
    in1d as in1d,
    isin as isin,
)

from numpy.lib.arrayterator import (
    Arrayterator as Arrayterator,
)

from numpy.lib.function_base import (
    select as select,
    piecewise as piecewise,
    trim_zeros as trim_zeros,
    copy as copy,
    iterable as iterable,
    percentile as percentile,
    diff as diff,
    gradient as gradient,
    angle as angle,
    unwrap as unwrap,
    sort_complex as sort_complex,
    disp as disp,
    flip as flip,
    rot90 as rot90,
    extract as extract,
    place as place,
    asarray_chkfinite as asarray_chkfinite,
    average as average,
    bincount as bincount,
    digitize as digitize,
    cov as cov,
    corrcoef as corrcoef,
    msort as msort,
    median as median,
    sinc as sinc,
    hamming as hamming,
    hanning as hanning,
    bartlett as bartlett,
    blackman as blackman,
    kaiser as kaiser,
    trapz as trapz,
    i0 as i0,
    add_newdoc as add_newdoc,
    add_docstring as add_docstring,
    meshgrid as meshgrid,
    delete as delete,
    insert as insert,
    append as append,
    interp as interp,
    add_newdoc_ufunc as add_newdoc_ufunc,
    quantile as quantile,
)

from numpy.lib.histograms import (
    histogram_bin_edges as histogram_bin_edges,
    histogram as histogram,
    histogramdd as histogramdd,
)

from numpy.lib.index_tricks import (
    ravel_multi_index as ravel_multi_index,
    unravel_index as unravel_index,
    mgrid as mgrid,
    ogrid as ogrid,
    r_ as r_,
    c_ as c_,
    s_ as s_,
    index_exp as index_exp,
    ix_ as ix_,
    fill_diagonal as fill_diagonal,
    diag_indices as diag_indices,
    diag_indices_from as diag_indices_from,
)

from numpy.lib.nanfunctions import (
    nansum as nansum,
    nanmax as nanmax,
    nanmin as nanmin,
    nanargmax as nanargmax,
    nanargmin as nanargmin,
    nanmean as nanmean,
    nanmedian as nanmedian,
    nanpercentile as nanpercentile,
    nanvar as nanvar,
    nanstd as nanstd,
    nanprod as nanprod,
    nancumsum as nancumsum,
    nancumprod as nancumprod,
    nanquantile as nanquantile,
)

from numpy.lib.npyio import (
    savetxt as savetxt,
    loadtxt as loadtxt,
    genfromtxt as genfromtxt,
    recfromtxt as recfromtxt,
    recfromcsv as recfromcsv,
    load as load,
    save as save,
    savez as savez,
    savez_compressed as savez_compressed,
    packbits as packbits,
    unpackbits as unpackbits,
    fromregex as fromregex,
)

from numpy.lib.polynomial import (
    poly as poly,
    roots as roots,
    polyint as polyint,
    polyder as polyder,
    polyadd as polyadd,
    polysub as polysub,
    polymul as polymul,
    polydiv as polydiv,
    polyval as polyval,
    polyfit as polyfit,
)

from numpy.lib.shape_base import (
    column_stack as column_stack,
    row_stack as row_stack,
    dstack as dstack,
    array_split as array_split,
    split as split,
    hsplit as hsplit,
    vsplit as vsplit,
    dsplit as dsplit,
    apply_over_axes as apply_over_axes,
    expand_dims as expand_dims,
    apply_along_axis as apply_along_axis,
    kron as kron,
    tile as tile,
    get_array_wrap as get_array_wrap,
    take_along_axis as take_along_axis,
    put_along_axis as put_along_axis,
)

from numpy.lib.stride_tricks import (
    broadcast_to as broadcast_to,
    broadcast_arrays as broadcast_arrays,
    broadcast_shapes as broadcast_shapes,
)

from numpy.lib.twodim_base import (
    diag as diag,
    diagflat as diagflat,
    eye as eye,
    fliplr as fliplr,
    flipud as flipud,
    tri as tri,
    triu as triu,
    tril as tril,
    vander as vander,
    histogram2d as histogram2d,
    mask_indices as mask_indices,
    tril_indices as tril_indices,
    tril_indices_from as tril_indices_from,
    triu_indices as triu_indices,
    triu_indices_from as triu_indices_from,
)

from numpy.lib.type_check import (
    mintypecode as mintypecode,
    asfarray as asfarray,
    real as real,
    imag as imag,
    iscomplex as iscomplex,
    isreal as isreal,
    iscomplexobj as iscomplexobj,
    isrealobj as isrealobj,
    nan_to_num as nan_to_num,
    real_if_close as real_if_close,
    typename as typename,
    common_type as common_type,
)

from numpy.lib.ufunclike import (
    fix as fix,
    isposinf as isposinf,
    isneginf as isneginf,
)

from numpy.lib.utils import (
    issubclass_ as issubclass_,
    issubsctype as issubsctype,
    issubdtype as issubdtype,
    deprecate as deprecate,
    deprecate_with_doc as deprecate_with_doc,
    get_include as get_include,
    info as info,
    source as source,
    who as who,
    lookfor as lookfor,
    byte_bounds as byte_bounds,
    safe_eval as safe_eval,
)

from numpy.matrixlib import (
    asmatrix as asmatrix,
    mat as mat,
    bmat as bmat,
)

_AnyStr_contra = TypeVar("_AnyStr_contra", str, bytes, contravariant=True)

# Protocol for representing file-like-objects accepted
# by `ndarray.tofile` and `fromfile`
class _IOProtocol(Protocol):
    def flush(self) -> object: ...
    def fileno(self) -> int: ...
    def tell(self) -> SupportsIndex: ...
    def seek(self, offset: int, whence: int, /) -> object: ...

# NOTE: `seek`, `write` and `flush` are technically only required
# for `readwrite`/`write` modes
class _MemMapIOProtocol(Protocol):
    def flush(self) -> object: ...
    def fileno(self) -> SupportsIndex: ...
    def tell(self) -> int: ...
    def seek(self, offset: int, whence: int, /) -> object: ...
    def write(self, s: bytes, /) -> object: ...
    @property
    def read(self) -> object: ...

class _SupportsWrite(Protocol[_AnyStr_contra]):
    def write(self, s: _AnyStr_contra, /) -> object: ...

__all__: List[str]
__path__: List[str]
__version__: str
__git_version__: str
test: PytestTester

# TODO: Move placeholders to their respective module once
# their annotations are properly implemented
#
# Placeholders for classes

# Some of these are aliases; others are wrappers with an identical signature
round = around
round_ = around
max = amax
min = amin
product = prod
cumproduct = cumprod
sometrue = any
alltrue = all

def show_config() -> None: ...

_NdArraySubClass = TypeVar("_NdArraySubClass", bound=ndarray)
_DTypeScalar_co = TypeVar("_DTypeScalar_co", covariant=True, bound=generic)
_ByteOrder = L["S", "<", ">", "=", "|", "L", "B", "N", "I"]

class dtype(Generic[_DTypeScalar_co]):
    names: None | Tuple[builtins.str, ...]
    # Overload for subclass of generic
    @overload
    def __new__(
        cls,
        dtype: Type[_DTypeScalar_co],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    # Overloads for string aliases, Python types, and some assorted
    # other special cases. Order is sometimes important because of the
    # subtype relationships
    #
    # bool < int < float < complex < object
    #
    # so we have to make sure the overloads for the narrowest type is
    # first.
    # Builtin types
    @overload
    def __new__(cls, dtype: Type[bool], align: bool = ..., copy: bool = ...) -> dtype[bool_]: ...
    @overload
    def __new__(cls, dtype: Type[int], align: bool = ..., copy: bool = ...) -> dtype[int_]: ...
    @overload
    def __new__(cls, dtype: None | Type[float], align: bool = ..., copy: bool = ...) -> dtype[float_]: ...
    @overload
    def __new__(cls, dtype: Type[complex], align: bool = ..., copy: bool = ...) -> dtype[complex_]: ...
    @overload
    def __new__(cls, dtype: Type[builtins.str], align: bool = ..., copy: bool = ...) -> dtype[str_]: ...
    @overload
    def __new__(cls, dtype: Type[bytes], align: bool = ..., copy: bool = ...) -> dtype[bytes_]: ...

    # `unsignedinteger` string-based representations and ctypes
    @overload
    def __new__(cls, dtype: _UInt8Codes | Type[ct.c_uint8], align: bool = ..., copy: bool = ...) -> dtype[uint8]: ...
    @overload
    def __new__(cls, dtype: _UInt16Codes | Type[ct.c_uint16], align: bool = ..., copy: bool = ...) -> dtype[uint16]: ...
    @overload
    def __new__(cls, dtype: _UInt32Codes | Type[ct.c_uint32], align: bool = ..., copy: bool = ...) -> dtype[uint32]: ...
    @overload
    def __new__(cls, dtype: _UInt64Codes | Type[ct.c_uint64], align: bool = ..., copy: bool = ...) -> dtype[uint64]: ...
    @overload
    def __new__(cls, dtype: _UByteCodes | Type[ct.c_ubyte], align: bool = ..., copy: bool = ...) -> dtype[ubyte]: ...
    @overload
    def __new__(cls, dtype: _UShortCodes | Type[ct.c_ushort], align: bool = ..., copy: bool = ...) -> dtype[ushort]: ...
    @overload
    def __new__(cls, dtype: _UIntCCodes | Type[ct.c_uint], align: bool = ..., copy: bool = ...) -> dtype[uintc]: ...

    # NOTE: We're assuming here that `uint_ptr_t == size_t`,
    # an assumption that does not hold in rare cases (same for `ssize_t`)
    @overload
    def __new__(cls, dtype: _UIntPCodes | Type[ct.c_void_p] | Type[ct.c_size_t], align: bool = ..., copy: bool = ...) -> dtype[uintp]: ...
    @overload
    def __new__(cls, dtype: _UIntCodes | Type[ct.c_ulong], align: bool = ..., copy: bool = ...) -> dtype[uint]: ...
    @overload
    def __new__(cls, dtype: _ULongLongCodes | Type[ct.c_ulonglong], align: bool = ..., copy: bool = ...) -> dtype[ulonglong]: ...

    # `signedinteger` string-based representations and ctypes
    @overload
    def __new__(cls, dtype: _Int8Codes | Type[ct.c_int8], align: bool = ..., copy: bool = ...) -> dtype[int8]: ...
    @overload
    def __new__(cls, dtype: _Int16Codes | Type[ct.c_int16], align: bool = ..., copy: bool = ...) -> dtype[int16]: ...
    @overload
    def __new__(cls, dtype: _Int32Codes | Type[ct.c_int32], align: bool = ..., copy: bool = ...) -> dtype[int32]: ...
    @overload
    def __new__(cls, dtype: _Int64Codes | Type[ct.c_int64], align: bool = ..., copy: bool = ...) -> dtype[int64]: ...
    @overload
    def __new__(cls, dtype: _ByteCodes | Type[ct.c_byte], align: bool = ..., copy: bool = ...) -> dtype[byte]: ...
    @overload
    def __new__(cls, dtype: _ShortCodes | Type[ct.c_short], align: bool = ..., copy: bool = ...) -> dtype[short]: ...
    @overload
    def __new__(cls, dtype: _IntCCodes | Type[ct.c_int], align: bool = ..., copy: bool = ...) -> dtype[intc]: ...
    @overload
    def __new__(cls, dtype: _IntPCodes | Type[ct.c_ssize_t], align: bool = ..., copy: bool = ...) -> dtype[intp]: ...
    @overload
    def __new__(cls, dtype: _IntCodes | Type[ct.c_long], align: bool = ..., copy: bool = ...) -> dtype[int_]: ...
    @overload
    def __new__(cls, dtype: _LongLongCodes | Type[ct.c_longlong], align: bool = ..., copy: bool = ...) -> dtype[longlong]: ...

    # `floating` string-based representations and ctypes
    @overload
    def __new__(cls, dtype: _Float16Codes, align: bool = ..., copy: bool = ...) -> dtype[float16]: ...
    @overload
    def __new__(cls, dtype: _Float32Codes, align: bool = ..., copy: bool = ...) -> dtype[float32]: ...
    @overload
    def __new__(cls, dtype: _Float64Codes, align: bool = ..., copy: bool = ...) -> dtype[float64]: ...
    @overload
    def __new__(cls, dtype: _HalfCodes, align: bool = ..., copy: bool = ...) -> dtype[half]: ...
    @overload
    def __new__(cls, dtype: _SingleCodes | Type[ct.c_float], align: bool = ..., copy: bool = ...) -> dtype[single]: ...
    @overload
    def __new__(cls, dtype: _DoubleCodes | Type[ct.c_double], align: bool = ..., copy: bool = ...) -> dtype[double]: ...
    @overload
    def __new__(cls, dtype: _LongDoubleCodes | Type[ct.c_longdouble], align: bool = ..., copy: bool = ...) -> dtype[longdouble]: ...

    # `complexfloating` string-based representations
    @overload
    def __new__(cls, dtype: _Complex64Codes, align: bool = ..., copy: bool = ...) -> dtype[complex64]: ...
    @overload
    def __new__(cls, dtype: _Complex128Codes, align: bool = ..., copy: bool = ...) -> dtype[complex128]: ...
    @overload
    def __new__(cls, dtype: _CSingleCodes, align: bool = ..., copy: bool = ...) -> dtype[csingle]: ...
    @overload
    def __new__(cls, dtype: _CDoubleCodes, align: bool = ..., copy: bool = ...) -> dtype[cdouble]: ...
    @overload
    def __new__(cls, dtype: _CLongDoubleCodes, align: bool = ..., copy: bool = ...) -> dtype[clongdouble]: ...

    # Miscellaneous string-based representations and ctypes
    @overload
    def __new__(cls, dtype: _BoolCodes | Type[ct.c_bool], align: bool = ..., copy: bool = ...) -> dtype[bool_]: ...
    @overload
    def __new__(cls, dtype: _TD64Codes, align: bool = ..., copy: bool = ...) -> dtype[timedelta64]: ...
    @overload
    def __new__(cls, dtype: _DT64Codes, align: bool = ..., copy: bool = ...) -> dtype[datetime64]: ...
    @overload
    def __new__(cls, dtype: _StrCodes, align: bool = ..., copy: bool = ...) -> dtype[str_]: ...
    @overload
    def __new__(cls, dtype: _BytesCodes | Type[ct.c_char], align: bool = ..., copy: bool = ...) -> dtype[bytes_]: ...
    @overload
    def __new__(cls, dtype: _VoidCodes, align: bool = ..., copy: bool = ...) -> dtype[void]: ...
    @overload
    def __new__(cls, dtype: _ObjectCodes | Type[ct.py_object], align: bool = ..., copy: bool = ...) -> dtype[object_]: ...

    # dtype of a dtype is the same dtype
    @overload
    def __new__(
        cls,
        dtype: dtype[_DTypeScalar_co],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    @overload
    def __new__(
        cls,
        dtype: _SupportsDType[dtype[_DTypeScalar_co]],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    # Handle strings that can't be expressed as literals; i.e. s1, s2, ...
    @overload
    def __new__(
        cls,
        dtype: builtins.str,
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[Any]: ...
    # Catchall overload for void-likes
    @overload
    def __new__(
        cls,
        dtype: _VoidDTypeLike,
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[void]: ...
    # Catchall overload for object-likes
    @overload
    def __new__(
        cls,
        dtype: Type[object],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[object_]: ...

    if sys.version_info >= (3, 9):
        def __class_getitem__(self, item: Any) -> GenericAlias: ...

    @overload
    def __getitem__(self: dtype[void], key: List[builtins.str]) -> dtype[void]: ...
    @overload
    def __getitem__(self: dtype[void], key: builtins.str | SupportsIndex) -> dtype[Any]: ...

    # NOTE: In the future 1-based multiplications will also yield `flexible` dtypes
    @overload
    def __mul__(self: _DType, value: L[1]) -> _DType: ...
    @overload
    def __mul__(self: _FlexDType, value: SupportsIndex) -> _FlexDType: ...
    @overload
    def __mul__(self, value: SupportsIndex) -> dtype[void]: ...

    # NOTE: `__rmul__` seems to be broken when used in combination with
    # literals as of mypy 0.902. Set the return-type to `dtype[Any]` for
    # now for non-flexible dtypes.
    @overload
    def __rmul__(self: _FlexDType, value: SupportsIndex) -> _FlexDType: ...
    @overload
    def __rmul__(self, value: SupportsIndex) -> dtype[Any]: ...

    def __gt__(self, other: DTypeLike) -> bool: ...
    def __ge__(self, other: DTypeLike) -> bool: ...
    def __lt__(self, other: DTypeLike) -> bool: ...
    def __le__(self, other: DTypeLike) -> bool: ...

    # Explicitly defined `__eq__` and `__ne__` to get around mypy's
    # `strict_equality` option; even though their signatures are
    # identical to their `object`-based counterpart
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...

    @property
    def alignment(self) -> int: ...
    @property
    def base(self) -> dtype[Any]: ...
    @property
    def byteorder(self) -> builtins.str: ...
    @property
    def char(self) -> builtins.str: ...
    @property
    def descr(self) -> List[Tuple[builtins.str, builtins.str] | Tuple[builtins.str, builtins.str, _Shape]]: ...
    @property
    def fields(
        self,
    ) -> None | MappingProxyType[builtins.str, Tuple[dtype[Any], int] | Tuple[dtype[Any], int, Any]]: ...
    @property
    def flags(self) -> int: ...
    @property
    def hasobject(self) -> bool: ...
    @property
    def isbuiltin(self) -> int: ...
    @property
    def isnative(self) -> bool: ...
    @property
    def isalignedstruct(self) -> bool: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def kind(self) -> builtins.str: ...
    @property
    def metadata(self) -> None | MappingProxyType[builtins.str, Any]: ...
    @property
    def name(self) -> builtins.str: ...
    @property
    def num(self) -> int: ...
    @property
    def shape(self) -> _Shape: ...
    @property
    def ndim(self) -> int: ...
    @property
    def subdtype(self) -> None | Tuple[dtype[Any], _Shape]: ...
    def newbyteorder(self: _DType, __new_order: _ByteOrder = ...) -> _DType: ...
    @property
    def str(self) -> builtins.str: ...
    @property
    def type(self) -> Type[_DTypeScalar_co]: ...

_ArrayLikeInt = Union[
    int,
    integer,
    Sequence[Union[int, integer]],
    Sequence[Sequence[Any]],  # TODO: wait for support for recursive types
    ndarray
]

_FlatIterSelf = TypeVar("_FlatIterSelf", bound=flatiter)

class flatiter(Generic[_NdArraySubClass]):
    @property
    def base(self) -> _NdArraySubClass: ...
    @property
    def coords(self) -> _Shape: ...
    @property
    def index(self) -> int: ...
    def copy(self) -> _NdArraySubClass: ...
    def __iter__(self: _FlatIterSelf) -> _FlatIterSelf: ...
    def __next__(self: flatiter[ndarray[Any, dtype[_ScalarType]]]) -> _ScalarType: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(
        self: flatiter[ndarray[Any, dtype[_ScalarType]]],
        key: int | integer | tuple[int | integer],
    ) -> _ScalarType: ...
    @overload
    def __getitem__(
        self,
        key: _ArrayLikeInt | slice | ellipsis | tuple[_ArrayLikeInt | slice | ellipsis],
    ) -> _NdArraySubClass: ...
    # TODO: `__setitem__` operates via `unsafe` casting rules, and can
    # thus accept any type accepted by the relevant underlying `np.generic`
    # constructor.
    # This means that `value` must in reality be a supertype of `npt.ArrayLike`.
    def __setitem__(
        self,
        key: _ArrayLikeInt | slice | ellipsis | tuple[_ArrayLikeInt | slice | ellipsis],
        value: Any,
    ) -> None: ...
    @overload
    def __array__(self: flatiter[ndarray[Any, _DType]], dtype: None = ..., /) -> ndarray[Any, _DType]: ...
    @overload
    def __array__(self, dtype: _DType, /) -> ndarray[Any, _DType]: ...

_OrderKACF = Optional[L["K", "A", "C", "F"]]
_OrderACF = Optional[L["A", "C", "F"]]
_OrderCF = Optional[L["C", "F"]]

_ModeKind = L["raise", "wrap", "clip"]
_PartitionKind = L["introselect"]
_SortKind = L["quicksort", "mergesort", "heapsort", "stable"]
_SortSide = L["left", "right"]

_ArraySelf = TypeVar("_ArraySelf", bound=_ArrayOrScalarCommon)

class _ArrayOrScalarCommon:
    @property
    def T(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def data(self) -> memoryview: ...
    @property
    def flags(self) -> flagsobj: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __copy__(self: _ArraySelf) -> _ArraySelf: ...
    def __deepcopy__(self: _ArraySelf, memo: None | Dict[int, Any], /) -> _ArraySelf: ...

    # TODO: How to deal with the non-commutative nature of `==` and `!=`?
    # xref numpy/numpy#17368
    def __eq__(self, other: Any) -> Any: ...
    def __ne__(self, other: Any) -> Any: ...
    def copy(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...
    def dump(self, file: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _SupportsWrite[bytes]) -> None: ...
    def dumps(self) -> bytes: ...
    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...
    # NOTE: `tostring()` is deprecated and therefore excluded
    # def tostring(self, order=...): ...
    def tofile(
        self,
        fid: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _IOProtocol,
        sep: str = ...,
        format: str = ...,
    ) -> None: ...
    # generics and 0d arrays return builtin scalars
    def tolist(self) -> Any: ...

    @property
    def __array_interface__(self) -> Dict[str, Any]: ...
    @property
    def __array_priority__(self) -> float: ...
    @property
    def __array_struct__(self) -> Any: ...  # builtins.PyCapsule
    def __setstate__(self, state: Tuple[
        SupportsIndex,  # version
        _ShapeLike,  # Shape
        _DType_co,  # DType
        bool,  # F-continuous
        bytes | List[Any],  # Data
    ], /) -> None: ...
    # a `bool_` is returned when `keepdims=True` and `self` is a 0d array

    @overload
    def all(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: L[False] = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> bool_: ...
    @overload
    def all(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def all(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def any(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: L[False] = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> bool_: ...
    @overload
    def any(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def any(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def argmax(
        self,
        axis: None = ...,
        out: None = ...,
        *,
        keepdims: L[False] = ...,
    ) -> intp: ...
    @overload
    def argmax(
        self,
        axis: _ShapeLike = ...,
        out: None = ...,
        *,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def argmax(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        *,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def argmin(
        self,
        axis: None = ...,
        out: None = ...,
        *,
        keepdims: L[False] = ...,
    ) -> intp: ...
    @overload
    def argmin(
        self,
        axis: _ShapeLike = ...,
        out: None = ...,
        *,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def argmin(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        *,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    def argsort(
        self,
        axis: Optional[SupportsIndex] = ...,
        kind: Optional[_SortKind] = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> ndarray: ...

    @overload
    def choose(
        self,
        choices: ArrayLike,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray: ...
    @overload
    def choose(
        self,
        choices: ArrayLike,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: Optional[ArrayLike] = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> ndarray: ...
    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> ndarray: ...
    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: Optional[ArrayLike] = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass: ...
    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass: ...

    @overload
    def compress(
        self,
        a: ArrayLike,
        axis: Optional[SupportsIndex] = ...,
        out: None = ...,
    ) -> ndarray: ...
    @overload
    def compress(
        self,
        a: ArrayLike,
        axis: Optional[SupportsIndex] = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    def conj(self: _ArraySelf) -> _ArraySelf: ...

    def conjugate(self: _ArraySelf) -> _ArraySelf: ...

    @overload
    def cumprod(
        self,
        axis: Optional[SupportsIndex] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> ndarray: ...
    @overload
    def cumprod(
        self,
        axis: Optional[SupportsIndex] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def cumsum(
        self,
        axis: Optional[SupportsIndex] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> ndarray: ...
    @overload
    def cumsum(
        self,
        axis: Optional[SupportsIndex] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def max(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def max(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def mean(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def mean(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def min(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def min(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    def newbyteorder(
        self: _ArraySelf,
        __new_order: _ByteOrder = ...,
    ) -> _ArraySelf: ...

    @overload
    def prod(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def prod(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def ptp(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def ptp(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def round(
        self: _ArraySelf,
        decimals: SupportsIndex = ...,
        out: None = ...,
    ) -> _ArraySelf: ...
    @overload
    def round(
        self,
        decimals: SupportsIndex = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def std(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def std(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: int = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def sum(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def sum(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def var(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def var(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: int = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

_DType = TypeVar("_DType", bound=dtype[Any])
_DType_co = TypeVar("_DType_co", covariant=True, bound=dtype[Any])
_FlexDType = TypeVar("_FlexDType", bound=dtype[flexible])

# TODO: Set the `bound` to something more suitable once we
# have proper shape support
_ShapeType = TypeVar("_ShapeType", bound=Any)
_ShapeType2 = TypeVar("_ShapeType2", bound=Any)
_NumberType = TypeVar("_NumberType", bound=number[Any])

# There is currently no exhaustive way to type the buffer protocol,
# as it is implemented exclusivelly in the C API (python/typing#593)
_SupportsBuffer = Union[
    bytes,
    bytearray,
    memoryview,
    _array.array[Any],
    mmap.mmap,
    NDArray[Any],
    generic,
]

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)
_2Tuple = Tuple[_T, _T]
_CastingKind = L["no", "equiv", "safe", "same_kind", "unsafe"]

_DTypeLike = Union[
    dtype[_ScalarType],
    Type[_ScalarType],
    _SupportsDType[dtype[_ScalarType]],
]

_ArrayUInt_co = NDArray[Union[bool_, unsignedinteger[Any]]]
_ArrayInt_co = NDArray[Union[bool_, integer[Any]]]
_ArrayFloat_co = NDArray[Union[bool_, integer[Any], floating[Any]]]
_ArrayComplex_co = NDArray[Union[bool_, integer[Any], floating[Any], complexfloating[Any, Any]]]
_ArrayNumber_co = NDArray[Union[bool_, number[Any]]]
_ArrayTD64_co = NDArray[Union[bool_, integer[Any], timedelta64]]

# Introduce an alias for `dtype` to avoid naming conflicts.
_dtype = dtype

# `builtins.PyCapsule` unfortunately lacks annotations as of the moment;
# use `Any` as a stopgap measure
_PyCapsule = Any

class _SupportsItem(Protocol[_T_co]):
    def item(self, args: Any, /) -> _T_co: ...

class _SupportsReal(Protocol[_T_co]):
    @property
    def real(self) -> _T_co: ...

class _SupportsImag(Protocol[_T_co]):
    @property
    def imag(self) -> _T_co: ...

class ndarray(_ArrayOrScalarCommon, Generic[_ShapeType, _DType_co]):
    @property
    def base(self) -> Optional[ndarray]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def real(
        self: ndarray[_ShapeType, dtype[_SupportsReal[_ScalarType]]],  # type: ignore[type-var]
    ) -> ndarray[_ShapeType, _dtype[_ScalarType]]: ...
    @real.setter
    def real(self, value: ArrayLike) -> None: ...
    @property
    def imag(
        self: ndarray[_ShapeType, dtype[_SupportsImag[_ScalarType]]],  # type: ignore[type-var]
    ) -> ndarray[_ShapeType, _dtype[_ScalarType]]: ...
    @imag.setter
    def imag(self, value: ArrayLike) -> None: ...
    def __new__(
        cls: Type[_ArraySelf],
        shape: _ShapeLike,
        dtype: DTypeLike = ...,
        buffer: None | _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: None | _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> _ArraySelf: ...

    if sys.version_info >= (3, 9):
        def __class_getitem__(self, item: Any) -> GenericAlias: ...

    @overload
    def __array__(self, dtype: None = ..., /) -> ndarray[Any, _DType_co]: ...
    @overload
    def __array__(self, dtype: _DType, /) -> ndarray[Any, _DType]: ...

    def __array_ufunc__(
        self,
        ufunc: ufunc,
        method: L["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any: ...

    __array_finalize__: Any

    def __array_wrap__(
        self,
        array: ndarray[_ShapeType2, _DType],
        context: None | Tuple[ufunc, Tuple[Any, ...], int] = ...,
        /,
    ) -> ndarray[_ShapeType2, _DType]: ...

    def __array_prepare__(
        self,
        array: ndarray[_ShapeType2, _DType],
        context: None | Tuple[ufunc, Tuple[Any, ...], int] = ...,
        /,
    ) -> ndarray[_ShapeType2, _DType]: ...

    @overload
    def __getitem__(self, key: Union[
        SupportsIndex,
        _ArrayLikeInt_co,
        Tuple[SupportsIndex | _ArrayLikeInt_co, ...],
    ]) -> Any: ...
    @overload
    def __getitem__(self, key: Union[
        None,
        slice,
        ellipsis,
        SupportsIndex,
        _ArrayLikeInt_co,
        Tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...],
    ]) -> ndarray[Any, _DType_co]: ...
    @overload
    def __getitem__(self: NDArray[void], key: str) -> NDArray[Any]: ...
    @overload
    def __getitem__(self: NDArray[void], key: list[str]) -> ndarray[_ShapeType, _dtype[void]]: ...

    @property
    def ctypes(self) -> _ctypes[int]: ...
    @property
    def shape(self) -> _Shape: ...
    @shape.setter
    def shape(self, value: _ShapeLike) -> None: ...
    @property
    def strides(self) -> _Shape: ...
    @strides.setter
    def strides(self, value: _ShapeLike) -> None: ...
    def byteswap(self: _ArraySelf, inplace: bool = ...) -> _ArraySelf: ...
    def fill(self, value: Any) -> None: ...
    @property
    def flat(self: _NdArraySubClass) -> flatiter[_NdArraySubClass]: ...

    # Use the same output type as that of the underlying `generic`
    @overload
    def item(
        self: ndarray[Any, _dtype[_SupportsItem[_T]]],  # type: ignore[type-var]
        *args: SupportsIndex,
    ) -> _T: ...
    @overload
    def item(
        self: ndarray[Any, _dtype[_SupportsItem[_T]]],  # type: ignore[type-var]
        args: Tuple[SupportsIndex, ...],
        /,
    ) -> _T: ...

    @overload
    def itemset(self, value: Any, /) -> None: ...
    @overload
    def itemset(self, item: _ShapeLike, value: Any, /) -> None: ...

    @overload
    def resize(self, new_shape: _ShapeLike, /, *, refcheck: bool = ...) -> None: ...
    @overload
    def resize(self, *new_shape: SupportsIndex, refcheck: bool = ...) -> None: ...

    def setflags(
        self, write: bool = ..., align: bool = ..., uic: bool = ...
    ) -> None: ...

    def squeeze(
        self,
        axis: Union[SupportsIndex, Tuple[SupportsIndex, ...]] = ...,
    ) -> ndarray[Any, _DType_co]: ...

    def swapaxes(
        self,
        axis1: SupportsIndex,
        axis2: SupportsIndex,
    ) -> ndarray[Any, _DType_co]: ...

    @overload
    def transpose(self: _ArraySelf, axes: _ShapeLike, /) -> _ArraySelf: ...
    @overload
    def transpose(self: _ArraySelf, *axes: SupportsIndex) -> _ArraySelf: ...

    def argpartition(
        self,
        kth: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
        kind: _PartitionKind = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> ndarray[Any, _dtype[intp]]: ...

    def diagonal(
        self,
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
    ) -> ndarray[Any, _DType_co]: ...

    # 1D + 1D returns a scalar;
    # all other with at least 1 non-0D array return an ndarray.
    @overload
    def dot(self, b: _ScalarLike_co, out: None = ...) -> ndarray: ...
    @overload
    def dot(self, b: ArrayLike, out: None = ...) -> Any: ...  # type: ignore[misc]
    @overload
    def dot(self, b: ArrayLike, out: _NdArraySubClass) -> _NdArraySubClass: ...

    # `nonzero()` is deprecated for 0d arrays/generics
    def nonzero(self) -> Tuple[ndarray[Any, _dtype[intp]], ...]: ...

    def partition(
        self,
        kth: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        kind: _PartitionKind = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> None: ...

    # `put` is technically available to `generic`,
    # but is pointless as `generic`s are immutable
    def put(
        self,
        ind: _ArrayLikeInt_co,
        v: ArrayLike,
        mode: _ModeKind = ...,
    ) -> None: ...

    @overload
    def searchsorted(  # type: ignore[misc]
        self,  # >= 1D array
        v: _ScalarLike_co,  # 0D array-like
        side: _SortSide = ...,
        sorter: Optional[_ArrayLikeInt_co] = ...,
    ) -> intp: ...
    @overload
    def searchsorted(
        self,  # >= 1D array
        v: ArrayLike,
        side: _SortSide = ...,
        sorter: Optional[_ArrayLikeInt_co] = ...,
    ) -> ndarray[Any, _dtype[intp]]: ...

    def setfield(
        self,
        val: ArrayLike,
        dtype: DTypeLike,
        offset: SupportsIndex = ...,
    ) -> None: ...

    def sort(
        self,
        axis: SupportsIndex = ...,
        kind: Optional[_SortKind] = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> None: ...

    @overload
    def trace(
        self,  # >= 2D array
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> Any: ...
    @overload
    def trace(
        self,  # >= 2D array
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def take(  # type: ignore[misc]
        self: ndarray[Any, _dtype[_ScalarType]],
        indices: _IntLike_co,
        axis: Optional[SupportsIndex] = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> _ScalarType: ...
    @overload
    def take(  # type: ignore[misc]
        self,
        indices: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...

    def repeat(
        self,
        repeats: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
    ) -> ndarray[Any, _DType_co]: ...

    def flatten(
        self,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _DType_co]: ...

    def ravel(
        self,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _DType_co]: ...

    @overload
    def reshape(
        self, shape: _ShapeLike, /, *, order: _OrderACF = ...
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def reshape(
        self, *shape: SupportsIndex, order: _OrderACF = ...
    ) -> ndarray[Any, _DType_co]: ...

    @overload
    def astype(
        self,
        dtype: _DTypeLike[_ScalarType],
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: bool = ...,
        copy: bool | _CopyMode = ...,
    ) -> NDArray[_ScalarType]: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: bool = ...,
        copy: bool | _CopyMode = ...,
    ) -> NDArray[Any]: ...

    @overload
    def view(self: _ArraySelf) -> _ArraySelf: ...
    @overload
    def view(self, type: Type[_NdArraySubClass]) -> _NdArraySubClass: ...
    @overload
    def view(self, dtype: _DTypeLike[_ScalarType]) -> NDArray[_ScalarType]: ...
    @overload
    def view(self, dtype: DTypeLike) -> NDArray[Any]: ...
    @overload
    def view(
        self,
        dtype: DTypeLike,
        type: Type[_NdArraySubClass],
    ) -> _NdArraySubClass: ...

    @overload
    def getfield(
        self,
        dtype: _DTypeLike[_ScalarType],
        offset: SupportsIndex = ...
    ) -> NDArray[_ScalarType]: ...
    @overload
    def getfield(
        self,
        dtype: DTypeLike,
        offset: SupportsIndex = ...
    ) -> NDArray[Any]: ...

    # Dispatch to the underlying `generic` via protocols
    def __int__(
        self: ndarray[Any, _dtype[SupportsInt]],  # type: ignore[type-var]
    ) -> int: ...

    def __float__(
        self: ndarray[Any, _dtype[SupportsFloat]],  # type: ignore[type-var]
    ) -> float: ...

    def __complex__(
        self: ndarray[Any, _dtype[SupportsComplex]],  # type: ignore[type-var]
    ) -> complex: ...

    def __index__(
        self: ndarray[Any, _dtype[SupportsIndex]],  # type: ignore[type-var]
    ) -> int: ...

    def __len__(self) -> int: ...
    def __setitem__(self, key, value): ...
    def __iter__(self) -> Any: ...
    def __contains__(self, key) -> bool: ...

    # The last overload is for catching recursive objects whose
    # nesting is too deep.
    # The first overload is for catching `bytes` (as they are a subtype of
    # `Sequence[int]`) and `str`. As `str` is a recursive sequence of
    # strings, it will pass through the final overload otherwise

    @overload
    def __lt__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...

    @overload
    def __le__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __le__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __le__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __le__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __le__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...

    @overload
    def __gt__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...

    @overload
    def __ge__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...

    # Unary ops
    @overload
    def __abs__(self: NDArray[bool_]) -> NDArray[bool_]: ...
    @overload
    def __abs__(self: NDArray[complexfloating[_NBit1, _NBit1]]) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __abs__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...
    @overload
    def __abs__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...
    @overload
    def __abs__(self: NDArray[object_]) -> Any: ...

    @overload
    def __invert__(self: NDArray[bool_]) -> NDArray[bool_]: ...
    @overload
    def __invert__(self: NDArray[_IntType]) -> NDArray[_IntType]: ...
    @overload
    def __invert__(self: NDArray[object_]) -> Any: ...

    @overload
    def __pos__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...
    @overload
    def __pos__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...
    @overload
    def __pos__(self: NDArray[object_]) -> Any: ...

    @overload
    def __neg__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...
    @overload
    def __neg__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...
    @overload
    def __neg__(self: NDArray[object_]) -> Any: ...

    # Binary ops
    # NOTE: `ndarray` does not implement `__imatmul__`
    @overload
    def __matmul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __matmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __matmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rmatmul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rmatmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmatmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __mod__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...
    @overload
    def __mod__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __mod__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rmod__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...
    @overload
    def __rmod__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmod__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __divmod__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> _2Tuple[NDArray[int8]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _2Tuple[NDArray[unsignedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _2Tuple[NDArray[signedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _2Tuple[NDArray[floating[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> Tuple[NDArray[int64], NDArray[timedelta64]]: ...

    @overload
    def __rdivmod__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> _2Tuple[NDArray[int8]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _2Tuple[NDArray[unsignedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _2Tuple[NDArray[signedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _2Tuple[NDArray[floating[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> Tuple[NDArray[int64], NDArray[timedelta64]]: ...

    @overload
    def __add__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> NDArray[datetime64]: ...
    @overload
    def __add__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __add__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __add__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __radd__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> NDArray[datetime64]: ...
    @overload
    def __radd__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __radd__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __radd__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __sub__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __sub__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __sub__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[timedelta64]: ...
    @overload
    def __sub__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __sub__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rsub__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __rsub__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> NDArray[datetime64]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[timedelta64]: ...
    @overload
    def __rsub__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rsub__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __mul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayTD64_co, other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __mul__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __mul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __mul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rmul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayTD64_co, other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __rmul__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __rmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __floordiv__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[int64]: ...
    @overload
    def __floordiv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __floordiv__(self: NDArray[timedelta64], other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __floordiv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __floordiv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rfloordiv__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[int64]: ...
    @overload
    def __rfloordiv__(self: NDArray[bool_], other: _ArrayLikeTD64_co) -> NoReturn: ...
    @overload
    def __rfloordiv__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __rfloordiv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rfloordiv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __pow__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __pow__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __pow__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rpow__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rpow__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rpow__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __truediv__(self: _ArrayInt_co, other: _ArrayInt_co) -> NDArray[float64]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[float64]: ...
    @overload
    def __truediv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __truediv__(self: NDArray[timedelta64], other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __truediv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __truediv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rtruediv__(self: _ArrayInt_co, other: _ArrayInt_co) -> NDArray[float64]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[float64]: ...
    @overload
    def __rtruediv__(self: NDArray[bool_], other: _ArrayLikeTD64_co) -> NoReturn: ...
    @overload
    def __rtruediv__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __rtruediv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rtruediv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __lshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __lshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __lshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __lshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __lshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rlshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rlshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rlshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rlshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rlshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rrshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rrshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rrshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rrshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rrshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __and__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __and__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __and__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __and__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __and__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rand__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __rand__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rand__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rand__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rand__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __xor__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __xor__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __xor__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __xor__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __xor__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rxor__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __rxor__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rxor__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rxor__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rxor__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __or__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __or__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __or__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __or__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __or__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __ror__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __ror__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __ror__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __ror__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __ror__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    # `np.generic` does not support inplace operations
    @overload
    def __iadd__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __iadd__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __iadd__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __iadd__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __iadd__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __iadd__(self: NDArray[timedelta64], other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __iadd__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __iadd__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __isub__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __isub__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __isub__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __isub__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __isub__(self: NDArray[timedelta64], other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __isub__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __isub__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __imul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __imul__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __imul__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __imul__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __imul__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __imul__(self: NDArray[timedelta64], other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __imul__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __itruediv__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __itruediv__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __itruediv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __itruediv__(self: NDArray[timedelta64], other: _ArrayLikeInt_co) -> NDArray[timedelta64]: ...
    @overload
    def __itruediv__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __ifloordiv__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ifloordiv__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ifloordiv__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __ifloordiv__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __ifloordiv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __ifloordiv__(self: NDArray[timedelta64], other: _ArrayLikeInt_co) -> NDArray[timedelta64]: ...
    @overload
    def __ifloordiv__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __ipow__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ipow__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ipow__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __ipow__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __ipow__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __imod__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __imod__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __imod__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __imod__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...
    @overload
    def __imod__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __ilshift__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ilshift__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ilshift__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __irshift__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __irshift__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __irshift__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __iand__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __iand__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __iand__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __iand__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __ixor__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __ixor__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ixor__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ixor__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __ior__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __ior__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ior__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ior__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    def __dlpack__(self: NDArray[number[Any]], *, stream: None = ...) -> _PyCapsule: ...
    def __dlpack_device__(self) -> Tuple[int, L[0]]: ...

    # Keep `dtype` at the bottom to avoid name conflicts with `np.dtype`
    @property
    def dtype(self) -> _DType_co: ...

# NOTE: while `np.generic` is not technically an instance of `ABCMeta`,
# the `@abstractmethod` decorator is herein used to (forcefully) deny
# the creation of `np.generic` instances.
# The `# type: ignore` comments are necessary to silence mypy errors regarding
# the missing `ABCMeta` metaclass.

# See https://github.com/numpy/numpy-stubs/pull/80 for more details.

_ScalarType = TypeVar("_ScalarType", bound=generic)
_NBit1 = TypeVar("_NBit1", bound=NBitBase)
_NBit2 = TypeVar("_NBit2", bound=NBitBase)

class generic(_ArrayOrScalarCommon):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def __array__(self: _ScalarType, dtype: None = ..., /) -> ndarray[Any, _dtype[_ScalarType]]: ...
    @overload
    def __array__(self, dtype: _DType, /) -> ndarray[Any, _DType]: ...
    @property
    def base(self) -> None: ...
    @property
    def ndim(self) -> L[0]: ...
    @property
    def size(self) -> L[1]: ...
    @property
    def shape(self) -> Tuple[()]: ...
    @property
    def strides(self) -> Tuple[()]: ...
    def byteswap(self: _ScalarType, inplace: L[False] = ...) -> _ScalarType: ...
    @property
    def flat(self: _ScalarType) -> flatiter[ndarray[Any, _dtype[_ScalarType]]]: ...

    @overload
    def astype(
        self,
        dtype: _DTypeLike[_ScalarType],
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: bool = ...,
        copy: bool | _CopyMode = ...,
    ) -> _ScalarType: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: bool = ...,
        copy: bool | _CopyMode = ...,
    ) -> Any: ...

    # NOTE: `view` will perform a 0D->scalar cast,
    # thus the array `type` is irrelevant to the output type
    @overload
    def view(
        self: _ScalarType,
        type: Type[ndarray[Any, Any]] = ...,
    ) -> _ScalarType: ...
    @overload
    def view(
        self,
        dtype: _DTypeLike[_ScalarType],
        type: Type[ndarray[Any, Any]] = ...,
    ) -> _ScalarType: ...
    @overload
    def view(
        self,
        dtype: DTypeLike,
        type: Type[ndarray[Any, Any]] = ...,
    ) -> Any: ...

    @overload
    def getfield(
        self,
        dtype: _DTypeLike[_ScalarType],
        offset: SupportsIndex = ...
    ) -> _ScalarType: ...
    @overload
    def getfield(
        self,
        dtype: DTypeLike,
        offset: SupportsIndex = ...
    ) -> Any: ...

    def item(
        self, args: L[0] | Tuple[()] | Tuple[L[0]] = ..., /,
    ) -> Any: ...

    @overload
    def take(  # type: ignore[misc]
        self: _ScalarType,
        indices: _IntLike_co,
        axis: Optional[SupportsIndex] = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> _ScalarType: ...
    @overload
    def take(  # type: ignore[misc]
        self: _ScalarType,
        indices: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...

    def repeat(
        self: _ScalarType,
        repeats: _ArrayLikeInt_co,
        axis: Optional[SupportsIndex] = ...,
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...

    def flatten(
        self: _ScalarType,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...

    def ravel(
        self: _ScalarType,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...

    @overload
    def reshape(
        self: _ScalarType, shape: _ShapeLike, /, *, order: _OrderACF = ...
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    @overload
    def reshape(
        self: _ScalarType, *shape: SupportsIndex, order: _OrderACF = ...
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...

    def squeeze(
        self: _ScalarType, axis: Union[L[0], Tuple[()]] = ...
    ) -> _ScalarType: ...
    def transpose(self: _ScalarType, axes: Tuple[()] = ..., /) -> _ScalarType: ...
    # Keep `dtype` at the bottom to avoid name conflicts with `np.dtype`
    @property
    def dtype(self: _ScalarType) -> _dtype[_ScalarType]: ...

class number(generic, Generic[_NBit1]):  # type: ignore
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    if sys.version_info >= (3, 9):
        def __class_getitem__(self, item: Any) -> GenericAlias: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    # Ensure that objects annotated as `number` support arithmetic operations
    __add__: _NumberOp
    __radd__: _NumberOp
    __sub__: _NumberOp
    __rsub__: _NumberOp
    __mul__: _NumberOp
    __rmul__: _NumberOp
    __floordiv__: _NumberOp
    __rfloordiv__: _NumberOp
    __pow__: _NumberOp
    __rpow__: _NumberOp
    __truediv__: _NumberOp
    __rtruediv__: _NumberOp
    __lt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __le__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __gt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __ge__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]

class bool_(generic):
    def __init__(self, value: object = ..., /) -> None: ...
    def item(
        self, args: L[0] | Tuple[()] | Tuple[L[0]] = ..., /,
    ) -> bool: ...
    def tolist(self) -> bool: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    __add__: _BoolOp[bool_]
    __radd__: _BoolOp[bool_]
    __sub__: _BoolSub
    __rsub__: _BoolSub
    __mul__: _BoolOp[bool_]
    __rmul__: _BoolOp[bool_]
    __floordiv__: _BoolOp[int8]
    __rfloordiv__: _BoolOp[int8]
    __pow__: _BoolOp[int8]
    __rpow__: _BoolOp[int8]
    __truediv__: _BoolTrueDiv
    __rtruediv__: _BoolTrueDiv
    def __invert__(self) -> bool_: ...
    __lshift__: _BoolBitOp[int8]
    __rlshift__: _BoolBitOp[int8]
    __rshift__: _BoolBitOp[int8]
    __rrshift__: _BoolBitOp[int8]
    __and__: _BoolBitOp[bool_]
    __rand__: _BoolBitOp[bool_]
    __xor__: _BoolBitOp[bool_]
    __rxor__: _BoolBitOp[bool_]
    __or__: _BoolBitOp[bool_]
    __ror__: _BoolBitOp[bool_]
    __mod__: _BoolMod
    __rmod__: _BoolMod
    __divmod__: _BoolDivMod
    __rdivmod__: _BoolDivMod
    __lt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __le__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __gt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __ge__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]

bool8 = bool_

class object_(generic):
    def __init__(self, value: object = ..., /) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    # The 3 protocols below may or may not raise,
    # depending on the underlying object
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...

object0 = object_

# The `datetime64` constructors requires an object with the three attributes below,
# and thus supports datetime duck typing
class _DatetimeScalar(Protocol):
    @property
    def day(self) -> int: ...
    @property
    def month(self) -> int: ...
    @property
    def year(self) -> int: ...

# TODO: `item`/`tolist` returns either `dt.date`, `dt.datetime` or `int`
# depending on the unit
class datetime64(generic):
    @overload
    def __init__(
        self,
        value: None | datetime64 | _CharLike_co | _DatetimeScalar = ...,
        format: _CharLike_co | Tuple[_CharLike_co, _IntLike_co] = ...,
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        value: int,
        format: _CharLike_co | Tuple[_CharLike_co, _IntLike_co],
        /,
    ) -> None: ...
    def __add__(self, other: _TD64Like_co) -> datetime64: ...
    def __radd__(self, other: _TD64Like_co) -> datetime64: ...
    @overload
    def __sub__(self, other: datetime64) -> timedelta64: ...
    @overload
    def __sub__(self, other: _TD64Like_co) -> datetime64: ...
    def __rsub__(self, other: datetime64) -> timedelta64: ...
    __lt__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __le__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __gt__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __ge__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]

_IntValue = Union[SupportsInt, _CharLike_co, SupportsIndex]
_FloatValue = Union[None, _CharLike_co, SupportsFloat, SupportsIndex]
_ComplexValue = Union[
    None,
    _CharLike_co,
    SupportsFloat,
    SupportsComplex,
    SupportsIndex,
    complex,  # `complex` is not a subtype of `SupportsComplex`
]

class integer(number[_NBit1]):  # type: ignore
    @property
    def numerator(self: _ScalarType) -> _ScalarType: ...
    @property
    def denominator(self) -> L[1]: ...
    @overload
    def __round__(self, ndigits: None = ...) -> int: ...
    @overload
    def __round__(self: _ScalarType, ndigits: SupportsIndex) -> _ScalarType: ...

    # NOTE: `__index__` is technically defined in the bottom-most
    # sub-classes (`int64`, `uint32`, etc)
    def item(
        self, args: L[0] | Tuple[()] | Tuple[L[0]] = ..., /,
    ) -> int: ...
    def tolist(self) -> int: ...
    def is_integer(self) -> L[True]: ...
    def bit_count(self: _ScalarType) -> int: ...
    def __index__(self) -> int: ...
    __truediv__: _IntTrueDiv[_NBit1]
    __rtruediv__: _IntTrueDiv[_NBit1]
    def __mod__(self, value: _IntLike_co) -> integer: ...
    def __rmod__(self, value: _IntLike_co) -> integer: ...
    def __invert__(self: _IntType) -> _IntType: ...
    # Ensure that objects annotated as `integer` support bit-wise operations
    def __lshift__(self, other: _IntLike_co) -> integer: ...
    def __rlshift__(self, other: _IntLike_co) -> integer: ...
    def __rshift__(self, other: _IntLike_co) -> integer: ...
    def __rrshift__(self, other: _IntLike_co) -> integer: ...
    def __and__(self, other: _IntLike_co) -> integer: ...
    def __rand__(self, other: _IntLike_co) -> integer: ...
    def __or__(self, other: _IntLike_co) -> integer: ...
    def __ror__(self, other: _IntLike_co) -> integer: ...
    def __xor__(self, other: _IntLike_co) -> integer: ...
    def __rxor__(self, other: _IntLike_co) -> integer: ...

class signedinteger(integer[_NBit1]):
    def __init__(self, value: _IntValue = ..., /) -> None: ...
    __add__: _SignedIntOp[_NBit1]
    __radd__: _SignedIntOp[_NBit1]
    __sub__: _SignedIntOp[_NBit1]
    __rsub__: _SignedIntOp[_NBit1]
    __mul__: _SignedIntOp[_NBit1]
    __rmul__: _SignedIntOp[_NBit1]
    __floordiv__: _SignedIntOp[_NBit1]
    __rfloordiv__: _SignedIntOp[_NBit1]
    __pow__: _SignedIntOp[_NBit1]
    __rpow__: _SignedIntOp[_NBit1]
    __lshift__: _SignedIntBitOp[_NBit1]
    __rlshift__: _SignedIntBitOp[_NBit1]
    __rshift__: _SignedIntBitOp[_NBit1]
    __rrshift__: _SignedIntBitOp[_NBit1]
    __and__: _SignedIntBitOp[_NBit1]
    __rand__: _SignedIntBitOp[_NBit1]
    __xor__: _SignedIntBitOp[_NBit1]
    __rxor__: _SignedIntBitOp[_NBit1]
    __or__: _SignedIntBitOp[_NBit1]
    __ror__: _SignedIntBitOp[_NBit1]
    __mod__: _SignedIntMod[_NBit1]
    __rmod__: _SignedIntMod[_NBit1]
    __divmod__: _SignedIntDivMod[_NBit1]
    __rdivmod__: _SignedIntDivMod[_NBit1]

int8 = signedinteger[_8Bit]
int16 = signedinteger[_16Bit]
int32 = signedinteger[_32Bit]
int64 = signedinteger[_64Bit]

byte = signedinteger[_NBitByte]
short = signedinteger[_NBitShort]
intc = signedinteger[_NBitIntC]
intp = signedinteger[_NBitIntP]
int0 = signedinteger[_NBitIntP]
int_ = signedinteger[_NBitInt]
longlong = signedinteger[_NBitLongLong]

# TODO: `item`/`tolist` returns either `dt.timedelta` or `int`
# depending on the unit
class timedelta64(generic):
    def __init__(
        self,
        value: None | int | _CharLike_co | dt.timedelta | timedelta64 = ...,
        format: _CharLike_co | Tuple[_CharLike_co, _IntLike_co] = ...,
        /,
    ) -> None: ...
    @property
    def numerator(self: _ScalarType) -> _ScalarType: ...
    @property
    def denominator(self) -> L[1]: ...

    # NOTE: Only a limited number of units support conversion
    # to builtin scalar types: `Y`, `M`, `ns`, `ps`, `fs`, `as`
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    def __add__(self, other: _TD64Like_co) -> timedelta64: ...
    def __radd__(self, other: _TD64Like_co) -> timedelta64: ...
    def __sub__(self, other: _TD64Like_co) -> timedelta64: ...
    def __rsub__(self, other: _TD64Like_co) -> timedelta64: ...
    def __mul__(self, other: _FloatLike_co) -> timedelta64: ...
    def __rmul__(self, other: _FloatLike_co) -> timedelta64: ...
    __truediv__: _TD64Div[float64]
    __floordiv__: _TD64Div[int64]
    def __rtruediv__(self, other: timedelta64) -> float64: ...
    def __rfloordiv__(self, other: timedelta64) -> int64: ...
    def __mod__(self, other: timedelta64) -> timedelta64: ...
    def __rmod__(self, other: timedelta64) -> timedelta64: ...
    def __divmod__(self, other: timedelta64) -> Tuple[int64, timedelta64]: ...
    def __rdivmod__(self, other: timedelta64) -> Tuple[int64, timedelta64]: ...
    __lt__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __le__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __gt__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __ge__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]

class unsignedinteger(integer[_NBit1]):
    # NOTE: `uint64 + signedinteger -> float64`
    def __init__(self, value: _IntValue = ..., /) -> None: ...
    __add__: _UnsignedIntOp[_NBit1]
    __radd__: _UnsignedIntOp[_NBit1]
    __sub__: _UnsignedIntOp[_NBit1]
    __rsub__: _UnsignedIntOp[_NBit1]
    __mul__: _UnsignedIntOp[_NBit1]
    __rmul__: _UnsignedIntOp[_NBit1]
    __floordiv__: _UnsignedIntOp[_NBit1]
    __rfloordiv__: _UnsignedIntOp[_NBit1]
    __pow__: _UnsignedIntOp[_NBit1]
    __rpow__: _UnsignedIntOp[_NBit1]
    __lshift__: _UnsignedIntBitOp[_NBit1]
    __rlshift__: _UnsignedIntBitOp[_NBit1]
    __rshift__: _UnsignedIntBitOp[_NBit1]
    __rrshift__: _UnsignedIntBitOp[_NBit1]
    __and__: _UnsignedIntBitOp[_NBit1]
    __rand__: _UnsignedIntBitOp[_NBit1]
    __xor__: _UnsignedIntBitOp[_NBit1]
    __rxor__: _UnsignedIntBitOp[_NBit1]
    __or__: _UnsignedIntBitOp[_NBit1]
    __ror__: _UnsignedIntBitOp[_NBit1]
    __mod__: _UnsignedIntMod[_NBit1]
    __rmod__: _UnsignedIntMod[_NBit1]
    __divmod__: _UnsignedIntDivMod[_NBit1]
    __rdivmod__: _UnsignedIntDivMod[_NBit1]

uint8 = unsignedinteger[_8Bit]
uint16 = unsignedinteger[_16Bit]
uint32 = unsignedinteger[_32Bit]
uint64 = unsignedinteger[_64Bit]

ubyte = unsignedinteger[_NBitByte]
ushort = unsignedinteger[_NBitShort]
uintc = unsignedinteger[_NBitIntC]
uintp = unsignedinteger[_NBitIntP]
uint0 = unsignedinteger[_NBitIntP]
uint = unsignedinteger[_NBitInt]
ulonglong = unsignedinteger[_NBitLongLong]

class inexact(number[_NBit1]):  # type: ignore
    def __getnewargs__(self: inexact[_64Bit]) -> Tuple[float, ...]: ...

_IntType = TypeVar("_IntType", bound=integer)
_FloatType = TypeVar('_FloatType', bound=floating)

class floating(inexact[_NBit1]):
    def __init__(self, value: _FloatValue = ..., /) -> None: ...
    def item(
        self, args: L[0] | Tuple[()] | Tuple[L[0]] = ...,
        /,
    ) -> float: ...
    def tolist(self) -> float: ...
    def is_integer(self) -> bool: ...
    def hex(self: float64) -> str: ...
    @classmethod
    def fromhex(cls: Type[float64], string: str, /) -> float64: ...
    def as_integer_ratio(self) -> Tuple[int, int]: ...
    if sys.version_info >= (3, 9):
        def __ceil__(self: float64) -> int: ...
        def __floor__(self: float64) -> int: ...
    def __trunc__(self: float64) -> int: ...
    def __getnewargs__(self: float64) -> Tuple[float]: ...
    def __getformat__(self: float64, typestr: L["double", "float"], /) -> str: ...
    @overload
    def __round__(self, ndigits: None = ...) -> int: ...
    @overload
    def __round__(self: _ScalarType, ndigits: SupportsIndex) -> _ScalarType: ...
    __add__: _FloatOp[_NBit1]
    __radd__: _FloatOp[_NBit1]
    __sub__: _FloatOp[_NBit1]
    __rsub__: _FloatOp[_NBit1]
    __mul__: _FloatOp[_NBit1]
    __rmul__: _FloatOp[_NBit1]
    __truediv__: _FloatOp[_NBit1]
    __rtruediv__: _FloatOp[_NBit1]
    __floordiv__: _FloatOp[_NBit1]
    __rfloordiv__: _FloatOp[_NBit1]
    __pow__: _FloatOp[_NBit1]
    __rpow__: _FloatOp[_NBit1]
    __mod__: _FloatMod[_NBit1]
    __rmod__: _FloatMod[_NBit1]
    __divmod__: _FloatDivMod[_NBit1]
    __rdivmod__: _FloatDivMod[_NBit1]

float16 = floating[_16Bit]
float32 = floating[_32Bit]
float64 = floating[_64Bit]

half = floating[_NBitHalf]
single = floating[_NBitSingle]
double = floating[_NBitDouble]
float_ = floating[_NBitDouble]
longdouble = floating[_NBitLongDouble]
longfloat = floating[_NBitLongDouble]

# The main reason for `complexfloating` having two typevars is cosmetic.
# It is used to clarify why `complex128`s precision is `_64Bit`, the latter
# describing the two 64 bit floats representing its real and imaginary component

class complexfloating(inexact[_NBit1], Generic[_NBit1, _NBit2]):
    def __init__(self, value: _ComplexValue = ..., /) -> None: ...
    def item(
        self, args: L[0] | Tuple[()] | Tuple[L[0]] = ..., /,
    ) -> complex: ...
    def tolist(self) -> complex: ...
    @property
    def real(self) -> floating[_NBit1]: ...  # type: ignore[override]
    @property
    def imag(self) -> floating[_NBit2]: ...  # type: ignore[override]
    def __abs__(self) -> floating[_NBit1]: ...  # type: ignore[override]
    def __getnewargs__(self: complex128) -> Tuple[float, float]: ...
    # NOTE: Deprecated
    # def __round__(self, ndigits=...): ...
    __add__: _ComplexOp[_NBit1]
    __radd__: _ComplexOp[_NBit1]
    __sub__: _ComplexOp[_NBit1]
    __rsub__: _ComplexOp[_NBit1]
    __mul__: _ComplexOp[_NBit1]
    __rmul__: _ComplexOp[_NBit1]
    __truediv__: _ComplexOp[_NBit1]
    __rtruediv__: _ComplexOp[_NBit1]
    __pow__: _ComplexOp[_NBit1]
    __rpow__: _ComplexOp[_NBit1]

complex64 = complexfloating[_32Bit, _32Bit]
complex128 = complexfloating[_64Bit, _64Bit]

csingle = complexfloating[_NBitSingle, _NBitSingle]
singlecomplex = complexfloating[_NBitSingle, _NBitSingle]
cdouble = complexfloating[_NBitDouble, _NBitDouble]
complex_ = complexfloating[_NBitDouble, _NBitDouble]
cfloat = complexfloating[_NBitDouble, _NBitDouble]
clongdouble = complexfloating[_NBitLongDouble, _NBitLongDouble]
clongfloat = complexfloating[_NBitLongDouble, _NBitLongDouble]
longcomplex = complexfloating[_NBitLongDouble, _NBitLongDouble]

class flexible(generic): ...  # type: ignore

# TODO: `item`/`tolist` returns either `bytes` or `tuple`
# depending on whether or not it's used as an opaque bytes sequence
# or a structure
class void(flexible):
    def __init__(self, value: _IntLike_co | bytes, /) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def setfield(
        self, val: ArrayLike, dtype: DTypeLike, offset: int = ...
    ) -> None: ...
    @overload
    def __getitem__(self, key: str | SupportsIndex) -> Any: ...
    @overload
    def __getitem__(self, key: list[str]) -> void: ...
    def __setitem__(
        self,
        key: str | List[str] | SupportsIndex,
        value: ArrayLike,
    ) -> None: ...

void0 = void

class character(flexible):  # type: ignore
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...

# NOTE: Most `np.bytes_` / `np.str_` methods return their
# builtin `bytes` / `str` counterpart

class bytes_(character, bytes):
    @overload
    def __init__(self, value: object = ..., /) -> None: ...
    @overload
    def __init__(
        self, value: str, /, encoding: str = ..., errors: str = ...
    ) -> None: ...
    def item(
        self, args: L[0] | Tuple[()] | Tuple[L[0]] = ..., /,
    ) -> bytes: ...
    def tolist(self) -> bytes: ...

string_ = bytes_
bytes0 = bytes_

class str_(character, str):
    @overload
    def __init__(self, value: object = ..., /) -> None: ...
    @overload
    def __init__(
        self, value: bytes, /, encoding: str = ..., errors: str = ...
    ) -> None: ...
    def item(
        self, args: L[0] | Tuple[()] | Tuple[L[0]] = ..., /,
    ) -> str: ...
    def tolist(self) -> str: ...

unicode_ = str_
str0 = str_

#
# Constants
#

Inf: Final[float]
Infinity: Final[float]
NAN: Final[float]
NINF: Final[float]
NZERO: Final[float]
NaN: Final[float]
PINF: Final[float]
PZERO: Final[float]
e: Final[float]
euler_gamma: Final[float]
inf: Final[float]
infty: Final[float]
nan: Final[float]
pi: Final[float]

CLIP: L[0]
WRAP: L[1]
RAISE: L[2]

ERR_IGNORE: L[0]
ERR_WARN: L[1]
ERR_RAISE: L[2]
ERR_CALL: L[3]
ERR_PRINT: L[4]
ERR_LOG: L[5]
ERR_DEFAULT: L[521]

SHIFT_DIVIDEBYZERO: L[0]
SHIFT_OVERFLOW: L[3]
SHIFT_UNDERFLOW: L[6]
SHIFT_INVALID: L[9]

FPE_DIVIDEBYZERO: L[1]
FPE_OVERFLOW: L[2]
FPE_UNDERFLOW: L[4]
FPE_INVALID: L[8]

FLOATING_POINT_SUPPORT: L[1]
UFUNC_BUFSIZE_DEFAULT = BUFSIZE

little_endian: Final[bool]
True_: Final[bool_]
False_: Final[bool_]

UFUNC_PYVALS_NAME: L["UFUNC_PYVALS"]

newaxis: None

# See `npt._ufunc` for more concrete nin-/nout-specific stubs
class ufunc:
    @property
    def __name__(self) -> str: ...
    @property
    def __doc__(self) -> str: ...
    __call__: Callable[..., Any]
    @property
    def nin(self) -> int: ...
    @property
    def nout(self) -> int: ...
    @property
    def nargs(self) -> int: ...
    @property
    def ntypes(self) -> int: ...
    @property
    def types(self) -> List[str]: ...
    # Broad return type because it has to encompass things like
    #
    # >>> np.logical_and.identity is True
    # True
    # >>> np.add.identity is 0
    # True
    # >>> np.sin.identity is None
    # True
    #
    # and any user-defined ufuncs.
    @property
    def identity(self) -> Any: ...
    # This is None for ufuncs and a string for gufuncs.
    @property
    def signature(self) -> Optional[str]: ...
    # The next four methods will always exist, but they will just
    # raise a ValueError ufuncs with that don't accept two input
    # arguments and return one output argument. Because of that we
    # can't type them very precisely.
    reduce: Any
    accumulate: Any
    reduce: Any
    outer: Any
    # Similarly at won't be defined for ufuncs that return multiple
    # outputs, so we can't type it very precisely.
    at: Any

# Parameters: `__name__`, `ntypes` and `identity`
absolute: _UFunc_Nin1_Nout1[L['absolute'], L[20], None]
add: _UFunc_Nin2_Nout1[L['add'], L[22], L[0]]
arccos: _UFunc_Nin1_Nout1[L['arccos'], L[8], None]
arccosh: _UFunc_Nin1_Nout1[L['arccosh'], L[8], None]
arcsin: _UFunc_Nin1_Nout1[L['arcsin'], L[8], None]
arcsinh: _UFunc_Nin1_Nout1[L['arcsinh'], L[8], None]
arctan2: _UFunc_Nin2_Nout1[L['arctan2'], L[5], None]
arctan: _UFunc_Nin1_Nout1[L['arctan'], L[8], None]
arctanh: _UFunc_Nin1_Nout1[L['arctanh'], L[8], None]
bitwise_and: _UFunc_Nin2_Nout1[L['bitwise_and'], L[12], L[-1]]
bitwise_not: _UFunc_Nin1_Nout1[L['invert'], L[12], None]
bitwise_or: _UFunc_Nin2_Nout1[L['bitwise_or'], L[12], L[0]]
bitwise_xor: _UFunc_Nin2_Nout1[L['bitwise_xor'], L[12], L[0]]
cbrt: _UFunc_Nin1_Nout1[L['cbrt'], L[5], None]
ceil: _UFunc_Nin1_Nout1[L['ceil'], L[7], None]
conj: _UFunc_Nin1_Nout1[L['conjugate'], L[18], None]
conjugate: _UFunc_Nin1_Nout1[L['conjugate'], L[18], None]
copysign: _UFunc_Nin2_Nout1[L['copysign'], L[4], None]
cos: _UFunc_Nin1_Nout1[L['cos'], L[9], None]
cosh: _UFunc_Nin1_Nout1[L['cosh'], L[8], None]
deg2rad: _UFunc_Nin1_Nout1[L['deg2rad'], L[5], None]
degrees: _UFunc_Nin1_Nout1[L['degrees'], L[5], None]
divide: _UFunc_Nin2_Nout1[L['true_divide'], L[11], None]
divmod: _UFunc_Nin2_Nout2[L['divmod'], L[15], None]
equal: _UFunc_Nin2_Nout1[L['equal'], L[23], None]
exp2: _UFunc_Nin1_Nout1[L['exp2'], L[8], None]
exp: _UFunc_Nin1_Nout1[L['exp'], L[10], None]
expm1: _UFunc_Nin1_Nout1[L['expm1'], L[8], None]
fabs: _UFunc_Nin1_Nout1[L['fabs'], L[5], None]
float_power: _UFunc_Nin2_Nout1[L['float_power'], L[4], None]
floor: _UFunc_Nin1_Nout1[L['floor'], L[7], None]
floor_divide: _UFunc_Nin2_Nout1[L['floor_divide'], L[21], None]
fmax: _UFunc_Nin2_Nout1[L['fmax'], L[21], None]
fmin: _UFunc_Nin2_Nout1[L['fmin'], L[21], None]
fmod: _UFunc_Nin2_Nout1[L['fmod'], L[15], None]
frexp: _UFunc_Nin1_Nout2[L['frexp'], L[4], None]
gcd: _UFunc_Nin2_Nout1[L['gcd'], L[11], L[0]]
greater: _UFunc_Nin2_Nout1[L['greater'], L[23], None]
greater_equal: _UFunc_Nin2_Nout1[L['greater_equal'], L[23], None]
heaviside: _UFunc_Nin2_Nout1[L['heaviside'], L[4], None]
hypot: _UFunc_Nin2_Nout1[L['hypot'], L[5], L[0]]
invert: _UFunc_Nin1_Nout1[L['invert'], L[12], None]
isfinite: _UFunc_Nin1_Nout1[L['isfinite'], L[20], None]
isinf: _UFunc_Nin1_Nout1[L['isinf'], L[20], None]
isnan: _UFunc_Nin1_Nout1[L['isnan'], L[20], None]
isnat: _UFunc_Nin1_Nout1[L['isnat'], L[2], None]
lcm: _UFunc_Nin2_Nout1[L['lcm'], L[11], None]
ldexp: _UFunc_Nin2_Nout1[L['ldexp'], L[8], None]
left_shift: _UFunc_Nin2_Nout1[L['left_shift'], L[11], None]
less: _UFunc_Nin2_Nout1[L['less'], L[23], None]
less_equal: _UFunc_Nin2_Nout1[L['less_equal'], L[23], None]
log10: _UFunc_Nin1_Nout1[L['log10'], L[8], None]
log1p: _UFunc_Nin1_Nout1[L['log1p'], L[8], None]
log2: _UFunc_Nin1_Nout1[L['log2'], L[8], None]
log: _UFunc_Nin1_Nout1[L['log'], L[10], None]
logaddexp2: _UFunc_Nin2_Nout1[L['logaddexp2'], L[4], float]
logaddexp: _UFunc_Nin2_Nout1[L['logaddexp'], L[4], float]
logical_and: _UFunc_Nin2_Nout1[L['logical_and'], L[20], L[True]]
logical_not: _UFunc_Nin1_Nout1[L['logical_not'], L[20], None]
logical_or: _UFunc_Nin2_Nout1[L['logical_or'], L[20], L[False]]
logical_xor: _UFunc_Nin2_Nout1[L['logical_xor'], L[19], L[False]]
matmul: _GUFunc_Nin2_Nout1[L['matmul'], L[19], None]
maximum: _UFunc_Nin2_Nout1[L['maximum'], L[21], None]
minimum: _UFunc_Nin2_Nout1[L['minimum'], L[21], None]
mod: _UFunc_Nin2_Nout1[L['remainder'], L[16], None]
modf: _UFunc_Nin1_Nout2[L['modf'], L[4], None]
multiply: _UFunc_Nin2_Nout1[L['multiply'], L[23], L[1]]
negative: _UFunc_Nin1_Nout1[L['negative'], L[19], None]
nextafter: _UFunc_Nin2_Nout1[L['nextafter'], L[4], None]
not_equal: _UFunc_Nin2_Nout1[L['not_equal'], L[23], None]
positive: _UFunc_Nin1_Nout1[L['positive'], L[19], None]
power: _UFunc_Nin2_Nout1[L['power'], L[18], None]
rad2deg: _UFunc_Nin1_Nout1[L['rad2deg'], L[5], None]
radians: _UFunc_Nin1_Nout1[L['radians'], L[5], None]
reciprocal: _UFunc_Nin1_Nout1[L['reciprocal'], L[18], None]
remainder: _UFunc_Nin2_Nout1[L['remainder'], L[16], None]
right_shift: _UFunc_Nin2_Nout1[L['right_shift'], L[11], None]
rint: _UFunc_Nin1_Nout1[L['rint'], L[10], None]
sign: _UFunc_Nin1_Nout1[L['sign'], L[19], None]
signbit: _UFunc_Nin1_Nout1[L['signbit'], L[4], None]
sin: _UFunc_Nin1_Nout1[L['sin'], L[9], None]
sinh: _UFunc_Nin1_Nout1[L['sinh'], L[8], None]
spacing: _UFunc_Nin1_Nout1[L['spacing'], L[4], None]
sqrt: _UFunc_Nin1_Nout1[L['sqrt'], L[10], None]
square: _UFunc_Nin1_Nout1[L['square'], L[18], None]
subtract: _UFunc_Nin2_Nout1[L['subtract'], L[21], None]
tan: _UFunc_Nin1_Nout1[L['tan'], L[8], None]
tanh: _UFunc_Nin1_Nout1[L['tanh'], L[8], None]
true_divide: _UFunc_Nin2_Nout1[L['true_divide'], L[11], None]
trunc: _UFunc_Nin1_Nout1[L['trunc'], L[7], None]

abs = absolute

class _CopyMode(enum.Enum):
    ALWAYS: L[True]
    IF_NEEDED: L[False]
    NEVER: L[2]

# Warnings
class ModuleDeprecationWarning(DeprecationWarning): ...
class VisibleDeprecationWarning(UserWarning): ...
class ComplexWarning(RuntimeWarning): ...
class RankWarning(UserWarning): ...

# Errors
class TooHardError(RuntimeError): ...

class AxisError(ValueError, IndexError):
    axis: None | int
    ndim: None | int
    @overload
    def __init__(self, axis: str, ndim: None = ..., msg_prefix: None = ...) -> None: ...
    @overload
    def __init__(self, axis: int, ndim: int, msg_prefix: None | str = ...) -> None: ...

_CallType = TypeVar("_CallType", bound=Union[_ErrFunc, _SupportsWrite[str]])

class errstate(Generic[_CallType], ContextDecorator):
    call: _CallType
    kwargs: _ErrDictOptional

    # Expand `**kwargs` into explicit keyword-only arguments
    def __init__(
        self,
        *,
        call: _CallType = ...,
        all: Optional[_ErrKind] = ...,
        divide: Optional[_ErrKind] = ...,
        over: Optional[_ErrKind] = ...,
        under: Optional[_ErrKind] = ...,
        invalid: Optional[_ErrKind] = ...,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
        /,
    ) -> None: ...

class ndenumerate(Generic[_ScalarType]):
    iter: flatiter[NDArray[_ScalarType]]
    @overload
    def __new__(
        cls, arr: _FiniteNestedSequence[_SupportsArray[dtype[_ScalarType]]],
    ) -> ndenumerate[_ScalarType]: ...
    @overload
    def __new__(cls, arr: str | _NestedSequence[str]) -> ndenumerate[str_]: ...
    @overload
    def __new__(cls, arr: bytes | _NestedSequence[bytes]) -> ndenumerate[bytes_]: ...
    @overload
    def __new__(cls, arr: bool | _NestedSequence[bool]) -> ndenumerate[bool_]: ...
    @overload
    def __new__(cls, arr: int | _NestedSequence[int]) -> ndenumerate[int_]: ...
    @overload
    def __new__(cls, arr: float | _NestedSequence[float]) -> ndenumerate[float_]: ...
    @overload
    def __new__(cls, arr: complex | _NestedSequence[complex]) -> ndenumerate[complex_]: ...
    def __next__(self: ndenumerate[_ScalarType]) -> Tuple[_Shape, _ScalarType]: ...
    def __iter__(self: _T) -> _T: ...

class ndindex:
    @overload
    def __init__(self, shape: tuple[SupportsIndex, ...], /) -> None: ...
    @overload
    def __init__(self, *shape: SupportsIndex) -> None: ...
    def __iter__(self: _T) -> _T: ...
    def __next__(self) -> _Shape: ...

class DataSource:
    def __init__(
        self,
        destpath: Union[None, str, os.PathLike[str]] = ...,
    ) -> None: ...
    def __del__(self) -> None: ...
    def abspath(self, path: str) -> str: ...
    def exists(self, path: str) -> bool: ...

    # Whether the file-object is opened in string or bytes mode (by default)
    # depends on the file-extension of `path`
    def open(
        self,
        path: str,
        mode: str = ...,
        encoding: Optional[str] = ...,
        newline: Optional[str] = ...,
    ) -> IO[Any]: ...

# TODO: The type of each `__next__` and `iters` return-type depends
# on the length and dtype of `args`; we can't describe this behavior yet
# as we lack variadics (PEP 646).
class broadcast:
    def __new__(cls, *args: ArrayLike) -> broadcast: ...
    @property
    def index(self) -> int: ...
    @property
    def iters(self) -> Tuple[flatiter[Any], ...]: ...
    @property
    def nd(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def numiter(self) -> int: ...
    @property
    def shape(self) -> _Shape: ...
    @property
    def size(self) -> int: ...
    def __next__(self) -> Tuple[Any, ...]: ...
    def __iter__(self: _T) -> _T: ...
    def reset(self) -> None: ...

class busdaycalendar:
    def __new__(
        cls,
        weekmask: ArrayLike = ...,
        holidays: ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    ) -> busdaycalendar: ...
    @property
    def weekmask(self) -> NDArray[bool_]: ...
    @property
    def holidays(self) -> NDArray[datetime64]: ...

class finfo(Generic[_FloatType]):
    dtype: dtype[_FloatType]
    bits: int
    eps: _FloatType
    epsneg: _FloatType
    iexp: int
    machep: int
    max: _FloatType
    maxexp: int
    min: _FloatType
    minexp: int
    negep: int
    nexp: int
    nmant: int
    precision: int
    resolution: _FloatType
    smallest_subnormal: _FloatType
    @property
    def smallest_normal(self) -> _FloatType: ...
    @property
    def tiny(self) -> _FloatType: ...
    @overload
    def __new__(
        cls, dtype: inexact[_NBit1] | _DTypeLike[inexact[_NBit1]]
    ) -> finfo[floating[_NBit1]]: ...
    @overload
    def __new__(
        cls, dtype: complex | float | Type[complex] | Type[float]
    ) -> finfo[float_]: ...
    @overload
    def __new__(
        cls, dtype: str
    ) -> finfo[floating[Any]]: ...

class iinfo(Generic[_IntType]):
    dtype: dtype[_IntType]
    kind: str
    bits: int
    key: str
    @property
    def min(self) -> int: ...
    @property
    def max(self) -> int: ...

    @overload
    def __new__(cls, dtype: _IntType | _DTypeLike[_IntType]) -> iinfo[_IntType]: ...
    @overload
    def __new__(cls, dtype: int | Type[int]) -> iinfo[int_]: ...
    @overload
    def __new__(cls, dtype: str) -> iinfo[Any]: ...

class format_parser:
    dtype: dtype[void]
    def __init__(
        self,
        formats: DTypeLike,
        names: None | str | Sequence[str],
        titles: None | str | Sequence[str],
        aligned: bool = ...,
        byteorder: None | _ByteOrder = ...,
    ) -> None: ...

class recarray(ndarray[_ShapeType, _DType_co]):
    # NOTE: While not strictly mandatory, we're demanding here that arguments
    # for the `format_parser`- and `dtype`-based dtype constructors are
    # mutually exclusive
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        dtype: None = ...,
        buf: None | _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: None | _ShapeLike = ...,
        *,
        formats: DTypeLike,
        names: None | str | Sequence[str] = ...,
        titles: None | str | Sequence[str] = ...,
        byteorder: None | _ByteOrder = ...,
        aligned: bool = ...,
        order: _OrderKACF = ...,
    ) -> recarray[Any, dtype[record]]: ...
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        dtype: DTypeLike,
        buf: None | _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: None | _ShapeLike = ...,
        formats: None = ...,
        names: None = ...,
        titles: None = ...,
        byteorder: None = ...,
        aligned: L[False] = ...,
        order: _OrderKACF = ...,
    ) -> recarray[Any, dtype[Any]]: ...
    def __array_finalize__(self, obj: object) -> None: ...
    def __getattribute__(self, attr: str) -> Any: ...
    def __setattr__(self, attr: str, val: ArrayLike) -> None: ...
    @overload
    def __getitem__(self, indx: Union[
        SupportsIndex,
        _ArrayLikeInt_co,
        Tuple[SupportsIndex | _ArrayLikeInt_co, ...],
    ]) -> Any: ...
    @overload
    def __getitem__(self: recarray[Any, dtype[void]], indx: Union[
        None,
        slice,
        ellipsis,
        SupportsIndex,
        _ArrayLikeInt_co,
        Tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...],
    ]) -> recarray[Any, _DType_co]: ...
    @overload
    def __getitem__(self, indx: Union[
        None,
        slice,
        ellipsis,
        SupportsIndex,
        _ArrayLikeInt_co,
        Tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...],
    ]) -> ndarray[Any, _DType_co]: ...
    @overload
    def __getitem__(self, indx: str) -> NDArray[Any]: ...
    @overload
    def __getitem__(self, indx: list[str]) -> recarray[_ShapeType, dtype[record]]: ...
    @overload
    def field(self, attr: int | str, val: None = ...) -> Any: ...
    @overload
    def field(self, attr: int | str, val: ArrayLike) -> None: ...

class record(void):
    def __getattribute__(self, attr: str) -> Any: ...
    def __setattr__(self, attr: str, val: ArrayLike) -> None: ...
    def pprint(self) -> str: ...
    @overload
    def __getitem__(self, key: str | SupportsIndex) -> Any: ...
    @overload
    def __getitem__(self, key: list[str]) -> record: ...

_NDIterFlagsKind = L[
    "buffered",
    "c_index",
    "copy_if_overlap",
    "common_dtype",
    "delay_bufalloc",
    "external_loop",
    "f_index",
    "grow_inner", "growinner",
    "multi_index",
    "ranged",
    "refs_ok",
    "reduce_ok",
    "zerosize_ok",
]

_NDIterOpFlagsKind = L[
    "aligned",
    "allocate",
    "arraymask",
    "copy",
    "config",
    "nbo",
    "no_subtype",
    "no_broadcast",
    "overlap_assume_elementwise",
    "readonly",
    "readwrite",
    "updateifcopy",
    "virtual",
    "writeonly",
    "writemasked"
]

@final
class nditer:
    def __new__(
        cls,
        op: ArrayLike | Sequence[ArrayLike],
        flags: None | Sequence[_NDIterFlagsKind] = ...,
        op_flags: None | Sequence[Sequence[_NDIterOpFlagsKind]] = ...,
        op_dtypes: DTypeLike | Sequence[DTypeLike] = ...,
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        op_axes: None | Sequence[Sequence[SupportsIndex]] = ...,
        itershape: None | _ShapeLike = ...,
        buffersize: SupportsIndex = ...,
    ) -> nditer: ...
    def __enter__(self) -> nditer: ...
    def __exit__(
        self,
        exc_type: None | Type[BaseException],
        exc_value: None | BaseException,
        traceback: None | TracebackType,
    ) -> None: ...
    def __iter__(self) -> nditer: ...
    def __next__(self) -> Tuple[NDArray[Any], ...]: ...
    def __len__(self) -> int: ...
    def __copy__(self) -> nditer: ...
    @overload
    def __getitem__(self, index: SupportsIndex) -> NDArray[Any]: ...
    @overload
    def __getitem__(self, index: slice) -> Tuple[NDArray[Any], ...]: ...
    def __setitem__(self, index: slice | SupportsIndex, value: ArrayLike) -> None: ...
    def close(self) -> None: ...
    def copy(self) -> nditer: ...
    def debug_print(self) -> None: ...
    def enable_external_loop(self) -> None: ...
    def iternext(self) -> bool: ...
    def remove_axis(self, i: SupportsIndex, /) -> None: ...
    def remove_multi_index(self) -> None: ...
    def reset(self) -> None: ...
    @property
    def dtypes(self) -> Tuple[dtype[Any], ...]: ...
    @property
    def finished(self) -> bool: ...
    @property
    def has_delayed_bufalloc(self) -> bool: ...
    @property
    def has_index(self) -> bool: ...
    @property
    def has_multi_index(self) -> bool: ...
    @property
    def index(self) -> int: ...
    @property
    def iterationneedsapi(self) -> bool: ...
    @property
    def iterindex(self) -> int: ...
    @property
    def iterrange(self) -> Tuple[int, ...]: ...
    @property
    def itersize(self) -> int: ...
    @property
    def itviews(self) -> Tuple[NDArray[Any], ...]: ...
    @property
    def multi_index(self) -> Tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def nop(self) -> int: ...
    @property
    def operands(self) -> Tuple[NDArray[Any], ...]: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def value(self) -> Tuple[NDArray[Any], ...]: ...

_MemMapModeKind = L[
    "readonly", "r",
    "copyonwrite", "c",
    "readwrite", "r+",
    "write", "w+",
]

class memmap(ndarray[_ShapeType, _DType_co]):
    __array_priority__: ClassVar[float]
    filename: str | None
    offset: int
    mode: str
    @overload
    def __new__(
        subtype,
        filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _MemMapIOProtocol,
        dtype: Type[uint8] = ...,
        mode: _MemMapModeKind = ...,
        offset: int = ...,
        shape: None | int | Tuple[int, ...] = ...,
        order: _OrderKACF = ...,
    ) -> memmap[Any, dtype[uint8]]: ...
    @overload
    def __new__(
        subtype,
        filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _MemMapIOProtocol,
        dtype: _DTypeLike[_ScalarType],
        mode: _MemMapModeKind = ...,
        offset: int = ...,
        shape: None | int | Tuple[int, ...] = ...,
        order: _OrderKACF = ...,
    ) -> memmap[Any, dtype[_ScalarType]]: ...
    @overload
    def __new__(
        subtype,
        filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _MemMapIOProtocol,
        dtype: DTypeLike,
        mode: _MemMapModeKind = ...,
        offset: int = ...,
        shape: None | int | Tuple[int, ...] = ...,
        order: _OrderKACF = ...,
    ) -> memmap[Any, dtype[Any]]: ...
    def __array_finalize__(self, obj: memmap[Any, Any]) -> None: ...
    def __array_wrap__(
        self,
        array: memmap[_ShapeType, _DType_co],
        context: None | Tuple[ufunc, Tuple[Any, ...], int] = ...,
    ) -> Any: ...
    def flush(self) -> None: ...

# TODO: Add a mypy plugin for managing functions whose output type is dependant
# on the literal value of some sort of signature (e.g. `einsum` and `vectorize`)
class vectorize:
    pyfunc: Callable[..., Any]
    cache: bool
    signature: None | str
    otypes: None | str
    excluded: Set[int | str]
    __doc__: None | str
    def __init__(
        self,
        pyfunc: Callable[..., Any],
        otypes: None | str | Iterable[DTypeLike] = ...,
        doc: None | str = ...,
        excluded: None | Iterable[int | str] = ...,
        cache: bool = ...,
        signature: None | str = ...,
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class poly1d:
    @property
    def variable(self) -> str: ...
    @property
    def order(self) -> int: ...
    @property
    def o(self) -> int: ...
    @property
    def roots(self) -> NDArray[Any]: ...
    @property
    def r(self) -> NDArray[Any]: ...

    @property
    def coeffs(self) -> NDArray[Any]: ...
    @coeffs.setter
    def coeffs(self, value: NDArray[Any]) -> None: ...

    @property
    def c(self) -> NDArray[Any]: ...
    @c.setter
    def c(self, value: NDArray[Any]) -> None: ...

    @property
    def coef(self) -> NDArray[Any]: ...
    @coef.setter
    def coef(self, value: NDArray[Any]) -> None: ...

    @property
    def coefficients(self) -> NDArray[Any]: ...
    @coefficients.setter
    def coefficients(self, value: NDArray[Any]) -> None: ...

    __hash__: None  # type: ignore

    @overload
    def __array__(self, t: None = ...) -> NDArray[Any]: ...
    @overload
    def __array__(self, t: _DType) -> ndarray[Any, _DType]: ...

    @overload
    def __call__(self, val: _ScalarLike_co) -> Any: ...
    @overload
    def __call__(self, val: poly1d) -> poly1d: ...
    @overload
    def __call__(self, val: ArrayLike) -> NDArray[Any]: ...

    def __init__(
        self,
        c_or_r: ArrayLike,
        r: bool = ...,
        variable: None | str = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __neg__(self) -> poly1d: ...
    def __pos__(self) -> poly1d: ...
    def __mul__(self, other: ArrayLike) -> poly1d: ...
    def __rmul__(self, other: ArrayLike) -> poly1d: ...
    def __add__(self, other: ArrayLike) -> poly1d: ...
    def __radd__(self, other: ArrayLike) -> poly1d: ...
    def __pow__(self, val: _FloatLike_co) -> poly1d: ...  # Integral floats are accepted
    def __sub__(self, other: ArrayLike) -> poly1d: ...
    def __rsub__(self, other: ArrayLike) -> poly1d: ...
    def __div__(self, other: ArrayLike) -> poly1d: ...
    def __truediv__(self, other: ArrayLike) -> poly1d: ...
    def __rdiv__(self, other: ArrayLike) -> poly1d: ...
    def __rtruediv__(self, other: ArrayLike) -> poly1d: ...
    def __getitem__(self, val: int) -> Any: ...
    def __setitem__(self, key: int, val: Any) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...
    def deriv(self, m: SupportsInt | SupportsIndex = ...) -> poly1d: ...
    def integ(
        self,
        m: SupportsInt | SupportsIndex = ...,
        k: None | _ArrayLikeComplex_co | _ArrayLikeObject_co = ...,
    ) -> poly1d: ...

class matrix(ndarray[_ShapeType, _DType_co]):
    __array_priority__: ClassVar[float]
    def __new__(
        subtype,
        data: ArrayLike,
        dtype: DTypeLike = ...,
        copy: bool = ...,
    ) -> matrix[Any, Any]: ...
    def __array_finalize__(self, obj: NDArray[Any]) -> None: ...

    @overload
    def __getitem__(self, key: Union[
        SupportsIndex,
        _ArrayLikeInt_co,
        Tuple[SupportsIndex | _ArrayLikeInt_co, ...],
    ]) -> Any: ...
    @overload
    def __getitem__(self, key: Union[
        None,
        slice,
        ellipsis,
        SupportsIndex,
        _ArrayLikeInt_co,
        Tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...],
    ]) -> matrix[Any, _DType_co]: ...
    @overload
    def __getitem__(self: NDArray[void], key: str) -> matrix[Any, dtype[Any]]: ...
    @overload
    def __getitem__(self: NDArray[void], key: list[str]) -> matrix[_ShapeType, dtype[void]]: ...

    def __mul__(self, other: ArrayLike) -> matrix[Any, Any]: ...
    def __rmul__(self, other: ArrayLike) -> matrix[Any, Any]: ...
    def __imul__(self, other: ArrayLike) -> matrix[_ShapeType, _DType_co]: ...
    def __pow__(self, other: ArrayLike) -> matrix[Any, Any]: ...
    def __ipow__(self, other: ArrayLike) -> matrix[_ShapeType, _DType_co]: ...

    @overload
    def sum(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ...) -> Any: ...
    @overload
    def sum(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ...) -> matrix[Any, Any]: ...
    @overload
    def sum(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def mean(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ...) -> Any: ...
    @overload
    def mean(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ...) -> matrix[Any, Any]: ...
    @overload
    def mean(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def std(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> Any: ...
    @overload
    def std(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> matrix[Any, Any]: ...
    @overload
    def std(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ..., ddof: float = ...) -> _NdArraySubClass: ...

    @overload
    def var(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> Any: ...
    @overload
    def var(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> matrix[Any, Any]: ...
    @overload
    def var(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ..., ddof: float = ...) -> _NdArraySubClass: ...

    @overload
    def prod(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ...) -> Any: ...
    @overload
    def prod(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ...) -> matrix[Any, Any]: ...
    @overload
    def prod(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def any(self, axis: None = ..., out: None = ...) -> bool_: ...
    @overload
    def any(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[bool_]]: ...
    @overload
    def any(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def all(self, axis: None = ..., out: None = ...) -> bool_: ...
    @overload
    def all(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[bool_]]: ...
    @overload
    def all(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def max(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> _ScalarType: ...
    @overload
    def max(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, _DType_co]: ...
    @overload
    def max(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def min(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> _ScalarType: ...
    @overload
    def min(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, _DType_co]: ...
    @overload
    def min(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def argmax(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> intp: ...
    @overload
    def argmax(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[intp]]: ...
    @overload
    def argmax(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def argmin(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> intp: ...
    @overload
    def argmin(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[intp]]: ...
    @overload
    def argmin(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def ptp(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> _ScalarType: ...
    @overload
    def ptp(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, _DType_co]: ...
    @overload
    def ptp(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    def squeeze(self, axis: None | _ShapeLike = ...) -> matrix[Any, _DType_co]: ...
    def tolist(self: matrix[Any, dtype[_SupportsItem[_T]]]) -> List[List[_T]]: ...  # type: ignore[typevar]
    def ravel(self, order: _OrderKACF = ...) -> matrix[Any, _DType_co]: ...
    def flatten(self, order: _OrderKACF = ...) -> matrix[Any, _DType_co]: ...

    @property
    def T(self) -> matrix[Any, _DType_co]: ...
    @property
    def I(self) -> matrix[Any, Any]: ...
    @property
    def A(self) -> ndarray[_ShapeType, _DType_co]: ...
    @property
    def A1(self) -> ndarray[Any, _DType_co]: ...
    @property
    def H(self) -> matrix[Any, _DType_co]: ...
    def getT(self) -> matrix[Any, _DType_co]: ...
    def getI(self) -> matrix[Any, Any]: ...
    def getA(self) -> ndarray[_ShapeType, _DType_co]: ...
    def getA1(self) -> ndarray[Any, _DType_co]: ...
    def getH(self) -> matrix[Any, _DType_co]: ...

_CharType = TypeVar("_CharType", str_, bytes_)
_CharDType = TypeVar("_CharDType", dtype[str_], dtype[bytes_])
_CharArray = chararray[Any, dtype[_CharType]]

class chararray(ndarray[_ShapeType, _CharDType]):
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        itemsize: SupportsIndex | SupportsInt = ...,
        unicode: L[False] = ...,
        buffer: _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> chararray[Any, dtype[bytes_]]: ...
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        itemsize: SupportsIndex | SupportsInt = ...,
        unicode: L[True] = ...,
        buffer: _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> chararray[Any, dtype[str_]]: ...

    def __array_finalize__(self, obj: NDArray[str_ | bytes_]) -> None: ...
    def __mul__(self, other: _ArrayLikeInt_co) -> chararray[Any, _CharDType]: ...
    def __rmul__(self, other: _ArrayLikeInt_co) -> chararray[Any, _CharDType]: ...
    def __mod__(self, i: Any) -> chararray[Any, _CharDType]: ...

    @overload
    def __eq__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __eq__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __ne__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __ne__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __ge__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __ge__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __le__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __le__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __gt__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __gt__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __lt__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __lt__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __add__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> _CharArray[str_]: ...
    @overload
    def __add__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> _CharArray[bytes_]: ...

    @overload
    def __radd__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> _CharArray[str_]: ...
    @overload
    def __radd__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> _CharArray[bytes_]: ...

    @overload
    def center(
        self: _CharArray[str_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def center(
        self: _CharArray[bytes_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def count(
        self: _CharArray[str_],
        sub: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def count(
        self: _CharArray[bytes_],
        sub: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...

    def decode(
        self: _CharArray[bytes_],
        encoding: None | str = ...,
        errors: None | str = ...,
    ) -> _CharArray[str_]: ...

    def encode(
        self: _CharArray[str_],
        encoding: None | str = ...,
        errors: None | str = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def endswith(
        self: _CharArray[str_],
        suffix: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[bool_]: ...
    @overload
    def endswith(
        self: _CharArray[bytes_],
        suffix: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[bool_]: ...

    def expandtabs(
        self,
        tabsize: _ArrayLikeInt_co = ...,
    ) -> chararray[Any, _CharDType]: ...

    @overload
    def find(
        self: _CharArray[str_],
        sub: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def find(
        self: _CharArray[bytes_],
        sub: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...

    @overload
    def index(
        self: _CharArray[str_],
        sub: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def index(
        self: _CharArray[bytes_],
        sub: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...

    @overload
    def join(
        self: _CharArray[str_],
        seq: _ArrayLikeStr_co,
    ) -> _CharArray[str_]: ...
    @overload
    def join(
        self: _CharArray[bytes_],
        seq: _ArrayLikeBytes_co,
    ) -> _CharArray[bytes_]: ...

    @overload
    def ljust(
        self: _CharArray[str_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def ljust(
        self: _CharArray[bytes_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def lstrip(
        self: _CharArray[str_],
        chars: None | _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def lstrip(
        self: _CharArray[bytes_],
        chars: None | _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def partition(
        self: _CharArray[str_],
        sep: _ArrayLikeStr_co,
    ) -> _CharArray[str_]: ...
    @overload
    def partition(
        self: _CharArray[bytes_],
        sep: _ArrayLikeBytes_co,
    ) -> _CharArray[bytes_]: ...

    @overload
    def replace(
        self: _CharArray[str_],
        old: _ArrayLikeStr_co,
        new: _ArrayLikeStr_co,
        count: None | _ArrayLikeInt_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def replace(
        self: _CharArray[bytes_],
        old: _ArrayLikeBytes_co,
        new: _ArrayLikeBytes_co,
        count: None | _ArrayLikeInt_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def rfind(
        self: _CharArray[str_],
        sub: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def rfind(
        self: _CharArray[bytes_],
        sub: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...

    @overload
    def rindex(
        self: _CharArray[str_],
        sub: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def rindex(
        self: _CharArray[bytes_],
        sub: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...

    @overload
    def rjust(
        self: _CharArray[str_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def rjust(
        self: _CharArray[bytes_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def rpartition(
        self: _CharArray[str_],
        sep: _ArrayLikeStr_co,
    ) -> _CharArray[str_]: ...
    @overload
    def rpartition(
        self: _CharArray[bytes_],
        sep: _ArrayLikeBytes_co,
    ) -> _CharArray[bytes_]: ...

    @overload
    def rsplit(
        self: _CharArray[str_],
        sep: None | _ArrayLikeStr_co = ...,
        maxsplit: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[object_]: ...
    @overload
    def rsplit(
        self: _CharArray[bytes_],
        sep: None | _ArrayLikeBytes_co = ...,
        maxsplit: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[object_]: ...

    @overload
    def rstrip(
        self: _CharArray[str_],
        chars: None | _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def rstrip(
        self: _CharArray[bytes_],
        chars: None | _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def split(
        self: _CharArray[str_],
        sep: None | _ArrayLikeStr_co = ...,
        maxsplit: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[object_]: ...
    @overload
    def split(
        self: _CharArray[bytes_],
        sep: None | _ArrayLikeBytes_co = ...,
        maxsplit: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[object_]: ...

    def splitlines(self, keepends: None | _ArrayLikeBool_co = ...) -> NDArray[object_]: ...

    @overload
    def startswith(
        self: _CharArray[str_],
        prefix: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[bool_]: ...
    @overload
    def startswith(
        self: _CharArray[bytes_],
        prefix: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[bool_]: ...

    @overload
    def strip(
        self: _CharArray[str_],
        chars: None | _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def strip(
        self: _CharArray[bytes_],
        chars: None | _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def translate(
        self: _CharArray[str_],
        table: _ArrayLikeStr_co,
        deletechars: None | _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def translate(
        self: _CharArray[bytes_],
        table: _ArrayLikeBytes_co,
        deletechars: None | _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    def zfill(self, width: _ArrayLikeInt_co) -> chararray[Any, _CharDType]: ...
    def capitalize(self) -> chararray[_ShapeType, _CharDType]: ...
    def title(self) -> chararray[_ShapeType, _CharDType]: ...
    def swapcase(self) -> chararray[_ShapeType, _CharDType]: ...
    def lower(self) -> chararray[_ShapeType, _CharDType]: ...
    def upper(self) -> chararray[_ShapeType, _CharDType]: ...
    def isalnum(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isalpha(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isdigit(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def islower(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isspace(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def istitle(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isupper(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isnumeric(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isdecimal(self) -> ndarray[_ShapeType, dtype[bool_]]: ...

# NOTE: Deprecated
# class MachAr: ...

class _SupportsDLPack(Protocol[_T_contra]):
    def __dlpack__(self, *, stream: None | _T_contra = ...) -> _PyCapsule: ...

def _from_dlpack(__obj: _SupportsDLPack[None]) -> NDArray[Any]: ...# Copyright (c) 2013 Google Inc. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.


import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation

from io import StringIO

from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax

generator_default_variables = {
    "EXECUTABLE_PREFIX": "",
    "EXECUTABLE_SUFFIX": "",
    "STATIC_LIB_PREFIX": "lib",
    "STATIC_LIB_SUFFIX": ".a",
    "SHARED_LIB_PREFIX": "lib",
    # Gyp expects the following variables to be expandable by the build
    # system to the appropriate locations.  Ninja prefers paths to be
    # known at gyp time.  To resolve this, introduce special
    # variables starting with $! and $| (which begin with a $ so gyp knows it
    # should be treated specially, but is otherwise an invalid
    # ninja/shell variable) that are passed to gyp here but expanded
    # before writing out into the target .ninja files; see
    # ExpandSpecial.
    # $! is used for variables that represent a path and that can only appear at
    # the start of a string, while $| is used for variables that can appear
    # anywhere in a string.
    "INTERMEDIATE_DIR": "$!INTERMEDIATE_DIR",
    "SHARED_INTERMEDIATE_DIR": "$!PRODUCT_DIR/gen",
    "PRODUCT_DIR": "$!PRODUCT_DIR",
    "CONFIGURATION_NAME": "$|CONFIGURATION_NAME",
    # Special variables that may be used by gyp 'rule' targets.
    # We generate definitions for these variables on the fly when processing a
    # rule.
    "RULE_INPUT_ROOT": "${root}",
    "RULE_INPUT_DIRNAME": "${dirname}",
    "RULE_INPUT_PATH": "${source}",
    "RULE_INPUT_EXT": "${ext}",
    "RULE_INPUT_NAME": "${name}",
}

# Placates pylint.
generator_additional_non_configuration_keys = []
generator_additional_path_sections = []
generator_extra_sources_for_rules = []
generator_filelist_paths = None

generator_supports_multiple_toolsets = gyp.common.CrossCompileRequested()


def StripPrefix(arg, prefix):
    if arg.startswith(prefix):
        return arg[len(prefix) :]
    return arg


def QuoteShellArgument(arg, flavor):
    """Quote a string such that it will be interpreted as a single argument
    by the shell."""
    # Rather than attempting to enumerate the bad shell characters, just
    # allow common OK ones and quote anything else.
    if re.match(r"^[a-zA-Z0-9_=.\\/-]+$", arg):
        return arg  # No quoting necessary.
    if flavor == "win":
        return gyp.msvs_emulation.QuoteForRspFile(arg)
    return "'" + arg.replace("'", "'" + '"\'"' + "'") + "'"


def Define(d, flavor):
    """Takes a preprocessor define and returns a -D parameter that's ninja- and
    shell-escaped."""
    if flavor == "win":
        # cl.exe replaces literal # characters with = in preprocessor definitions for
        # some reason. Octal-encode to work around that.
        d = d.replace("#", "\\%03o" % ord("#"))
    return QuoteShellArgument(ninja_syntax.escape("-D" + d), flavor)


def AddArch(output, arch):
    """Adds an arch string to an output path."""
    output, extension = os.path.splitext(output)
    return f"{output}.{arch}{extension}"


class Target:
    """Target represents the paths used within a single gyp target.

    Conceptually, building a single target A is a series of steps:

    1) actions/rules/copies  generates source/resources/etc.
    2) compiles              generates .o files
    3) link                  generates a binary (library/executable)
    4) bundle                merges the above in a mac bundle

    (Any of these steps can be optional.)

    From a build ordering perspective, a dependent target B could just
    depend on the last output of this series of steps.

    But some dependent commands sometimes need to reach inside the box.
    For example, when linking B it needs to get the path to the static
    library generated by A.

    This object stores those paths.  To keep things simple, member
    variables only store concrete paths to single files, while methods
    compute derived values like "the last output of the target".
    """

    def __init__(self, type):
        # Gyp type ("static_library", etc.) of this target.
        self.type = type
        # File representing whether any input dependencies necessary for
        # dependent actions have completed.
        self.preaction_stamp = None
        # File representing whether any input dependencies necessary for
        # dependent compiles have completed.
        self.precompile_stamp = None
        # File representing the completion of actions/rules/copies, if any.
        self.actions_stamp = None
        # Path to the output of the link step, if any.
        self.binary = None
        # Path to the file representing the completion of building the bundle,
        # if any.
        self.bundle = None
        # On Windows, incremental linking requires linking against all the .objs
        # that compose a .lib (rather than the .lib itself). That list is stored
        # here. In this case, we also need to save the compile_deps for the target,
        # so that the target that directly depends on the .objs can also depend
        # on those.
        self.component_objs = None
        self.compile_deps = None
        # Windows only. The import .lib is the output of a build step, but
        # because dependents only link against the lib (not both the lib and the
        # dll) we keep track of the import library here.
        self.import_lib = None
        # Track if this target contains any C++ files, to decide if gcc or g++
        # should be used for linking.
        self.uses_cpp = False

    def Linkable(self):
        """Return true if this is a target that can be linked against."""
        return self.type in ("static_library", "shared_library")

    def UsesToc(self, flavor):
        """Return true if the target should produce a restat rule based on a TOC
        file."""
        # For bundles, the .TOC should be produced for the binary, not for
        # FinalOutput(). But the naive approach would put the TOC file into the
        # bundle, so don't do this for bundles for now.
        if flavor == "win" or self.bundle:
            return False
        return self.type in ("shared_library", "loadable_module")

    def PreActionInput(self, flavor):
        """Return the path, if any, that should be used as a dependency of
        any dependent action step."""
        if self.UsesToc(flavor):
            return self.FinalOutput() + ".TOC"
        return self.FinalOutput() or self.preaction_stamp

    def PreCompileInput(self):
        """Return the path, if any, that should be used as a dependency of
        any dependent compile step."""
        return self.actions_stamp or self.precompile_stamp

    def FinalOutput(self):
        """Return the last output of the target, which depends on all prior
        steps."""
        return self.bundle or self.binary or self.actions_stamp


# A small discourse on paths as used within the Ninja build:
# All files we produce (both at gyp and at build time) appear in the
# build directory (e.g. out/Debug).
#
# Paths within a given .gyp file are always relative to the directory
# containing the .gyp file.  Call these "gyp paths".  This includes
# sources as well as the starting directory a given gyp rule/action
# expects to be run from.  We call the path from the source root to
# the gyp file the "base directory" within the per-.gyp-file
# NinjaWriter code.
#
# All paths as written into the .ninja files are relative to the build
# directory.  Call these paths "ninja paths".
#
# We translate between these two notions of paths with two helper
# functions:
#
# - GypPathToNinja translates a gyp path (i.e. relative to the .gyp file)
#   into the equivalent ninja path.
#
# - GypPathToUniqueOutput translates a gyp path into a ninja path to write
#   an output file; the result can be namespaced such that it is unique
#   to the input file name as well as the output target name.


class NinjaWriter:
    def __init__(
        self,
        hash_for_rules,
        target_outputs,
        base_dir,
        build_dir,
        output_file,
        toplevel_build,
        output_file_name,
        flavor,
        toplevel_dir=None,
    ):
        """
        base_dir: path from source root to directory containing this gyp file,
                  by gyp semantics, all input paths are relative to this
        build_dir: path from source root to build output
        toplevel_dir: path to the toplevel directory
        """

        self.hash_for_rules = hash_for_rules
        self.target_outputs = target_outputs
        self.base_dir = base_dir
        self.build_dir = build_dir
        self.ninja = ninja_syntax.Writer(output_file)
        self.toplevel_build = toplevel_build
        self.output_file_name = output_file_name

        self.flavor = flavor
        self.abs_build_dir = None
        if toplevel_dir is not None:
            self.abs_build_dir = os.path.abspath(os.path.join(toplevel_dir, build_dir))
        self.obj_ext = ".obj" if flavor == "win" else ".o"
        if flavor == "win":
            # See docstring of msvs_emulation.GenerateEnvironmentFiles().
            self.win_env = {}
            for arch in ("x86", "x64"):
                self.win_env[arch] = "environment." + arch

        # Relative path from build output dir to base dir.
        build_to_top = gyp.common.InvertRelativePath(build_dir, toplevel_dir)
        self.build_to_base = os.path.join(build_to_top, base_dir)
        # Relative path from base dir to build dir.
        base_to_top = gyp.common.InvertRelativePath(base_dir, toplevel_dir)
        self.base_to_build = os.path.join(base_to_top, build_dir)

    def ExpandSpecial(self, path, product_dir=None):
        """Expand specials like $!PRODUCT_DIR in |path|.

        If |product_dir| is None, assumes the cwd is already the product
        dir.  Otherwise, |product_dir| is the relative path to the product
        dir.
        """

        PRODUCT_DIR = "$!PRODUCT_DIR"
        if PRODUCT_DIR in path:
            if product_dir:
                path = path.replace(PRODUCT_DIR, product_dir)
            else:
                path = path.replace(PRODUCT_DIR + "/", "")
                path = path.replace(PRODUCT_DIR + "\\", "")
                path = path.replace(PRODUCT_DIR, ".")

        INTERMEDIATE_DIR = "$!INTERMEDIATE_DIR"
        if INTERMEDIATE_DIR in path:
            int_dir = self.GypPathToUniqueOutput("gen")
            # GypPathToUniqueOutput generates a path relative to the product dir,
            # so insert product_dir in front if it is provided.
            path = path.replace(
                INTERMEDIATE_DIR, os.path.join(product_dir or "", int_dir)
            )

        CONFIGURATION_NAME = "$|CONFIGURATION_NAME"
        path = path.replace(CONFIGURATION_NAME, self.config_name)

        return path

    def ExpandRuleVariables(self, path, root, dirname, source, ext, name):
        if self.flavor == "win":
            path = self.msvs_settings.ConvertVSMacros(path, config=self.config_name)
        path = path.replace(generator_default_variables["RULE_INPUT_ROOT"], root)
        path = path.replace(generator_default_variables["RULE_INPUT_DIRNAME"], dirname)
        path = path.replace(generator_default_variables["RULE_INPUT_PATH"], source)
        path = path.replace(generator_default_variables["RULE_INPUT_EXT"], ext)
        path = path.replace(generator_default_variables["RULE_INPUT_NAME"], name)
        return path

    def GypPathToNinja(self, path, env=None):
        """Translate a gyp path to a ninja path, optionally expanding environment
        variable references in |path| with |env|.

        See the above discourse on path conversions."""
        if env:
            if self.flavor == "mac":
                path = gyp.xcode_emulation.ExpandEnvVars(path, env)
            elif self.flavor == "win":
                path = gyp.msvs_emulation.ExpandMacros(path, env)
        if path.startswith("$!"):
            expanded = self.ExpandSpecial(path)
            if self.flavor == "win":
                expanded = os.path.normpath(expanded)
            return expanded
        if "$|" in path:
            path = self.ExpandSpecial(path)
        assert "$" not in path, path
        return os.path.normpath(os.path.join(self.build_to_base, path))

    def GypPathToUniqueOutput(self, path, qualified=True):
        """Translate a gyp path to a ninja path for writing output.

        If qualified is True, qualify the resulting filename with the name
        of the target.  This is necessary when e.g. compiling the same
        path twice for two separate output targets.

        See the above discourse on path conversions."""

        path = self.ExpandSpecial(path)
        assert not path.startswith("$"), path

        # Translate the path following this scheme:
        #   Input: foo/bar.gyp, target targ, references baz/out.o
        #   Output: obj/foo/baz/targ.out.o (if qualified)
        #           obj/foo/baz/out.o (otherwise)
        #     (and obj.host instead of obj for cross-compiles)
        #
        # Why this scheme and not some other one?
        # 1) for a given input, you can compute all derived outputs by matching
        #    its path, even if the input is brought via a gyp file with '..'.
        # 2) simple files like libraries and stamps have a simple filename.

        obj = "obj"
        if self.toolset != "target":
            obj += "." + self.toolset

        path_dir, path_basename = os.path.split(path)
        assert not os.path.isabs(path_dir), (
            "'%s' can not be absolute path (see crbug.com/462153)." % path_dir
        )

        if qualified:
            path_basename = self.name + "." + path_basename
        return os.path.normpath(
            os.path.join(obj, self.base_dir, path_dir, path_basename)
        )

    def WriteCollapsedDependencies(self, name, targets, order_only=None):
        """Given a list of targets, return a path for a single file
        representing the result of building all the targets or None.

        Uses a stamp file if necessary."""

        assert targets == [item for item in targets if item], targets
        if len(targets) == 0:
            assert not order_only
            return None
        if len(targets) > 1 or order_only:
            stamp = self.GypPathToUniqueOutput(name + ".stamp")
            targets = self.ninja.build(stamp, "stamp", targets, order_only=order_only)
            self.ninja.newline()
        return targets[0]

    def _SubninjaNameForArch(self, arch):
        output_file_base = os.path.splitext(self.output_file_name)[0]
        return f"{output_file_base}.{arch}.ninja"

    def WriteSpec(self, spec, config_name, generator_flags):
        """The main entry point for NinjaWriter: write the build rules for a spec.

        Returns a Target object, which represents the output paths for this spec.
        Returns None if there are no outputs (e.g. a settings-only 'none' type
        target)."""

        self.config_name = config_name
        self.name = spec["target_name"]
        self.toolset = spec["toolset"]
        config = spec["configurations"][config_name]
        self.target = Target(spec["type"])
        self.is_standalone_static_library = bool(
            spec.get("standalone_static_library", 0)
        )

        self.target_rpath = generator_flags.get("target_rpath", r"\$$ORIGIN/lib/")

        self.is_mac_bundle = gyp.xcode_emulation.IsMacBundle(self.flavor, spec)
        self.xcode_settings = self.msvs_settings = None
        if self.flavor == "mac":
            self.xcode_settings = gyp.xcode_emulation.XcodeSettings(spec)
            mac_toolchain_dir = generator_flags.get("mac_toolchain_dir", None)
            if mac_toolchain_dir:
                self.xcode_settings.mac_toolchain_dir = mac_toolchain_dir

        if self.flavor == "win":
            self.msvs_settings = gyp.msvs_emulation.MsvsSettings(spec, generator_flags)
            arch = self.msvs_settings.GetArch(config_name)
            self.ninja.variable("arch", self.win_env[arch])
            self.ninja.variable("cc", "$cl_" + arch)
            self.ninja.variable("cxx", "$cl_" + arch)
            self.ninja.variable("cc_host", "$cl_" + arch)
            self.ninja.variable("cxx_host", "$cl_" + arch)
            self.ninja.variable("asm", "$ml_" + arch)

        if self.flavor == "mac":
            self.archs = self.xcode_settings.GetActiveArchs(config_name)
            if len(self.archs) > 1:
                self.arch_subninjas = {
                    arch: ninja_syntax.Writer(
                        OpenOutput(
                            os.path.join(
                                self.toplevel_build, self._SubninjaNameForArch(arch)
                            ),
                            "w",
                        )
                    )
                    for arch in self.archs
                }

        # Compute predepends for all rules.
        # actions_depends is the dependencies this target depends on before running
        # any of its action/rule/copy steps.
        # compile_depends is the dependencies this target depends on before running
        # any of its compile steps.
        actions_depends = []
        compile_depends = []
        # TODO(evan): it is rather confusing which things are lists and which
        # are strings.  Fix these.
        if "dependencies" in spec:
            for dep in spec["dependencies"]:
                if dep in self.target_outputs:
                    target = self.target_outputs[dep]
                    actions_depends.append(target.PreActionInput(self.flavor))
                    compile_depends.append(target.PreCompileInput())
                    if target.uses_cpp:
                        self.target.uses_cpp = True
            actions_depends = [item for item in actions_depends if item]
            compile_depends = [item for item in compile_depends if item]
            actions_depends = self.WriteCollapsedDependencies(
                "actions_depends", actions_depends
            )
            compile_depends = self.WriteCollapsedDependencies(
                "compile_depends", compile_depends
            )
            self.target.preaction_stamp = actions_depends
            self.target.precompile_stamp = compile_depends

        # Write out actions, rules, and copies.  These must happen before we
        # compile any sources, so compute a list of predependencies for sources
        # while we do it.
        extra_sources = []
        mac_bundle_depends = []
        self.target.actions_stamp = self.WriteActionsRulesCopies(
            spec, extra_sources, actions_depends, mac_bundle_depends
        )

        # If we have actions/rules/copies, we depend directly on those, but
        # otherwise we depend on dependent target's actions/rules/copies etc.
        # We never need to explicitly depend on previous target's link steps,
        # because no compile ever depends on them.
        compile_depends_stamp = self.target.actions_stamp or compile_depends

        # Write out the compilation steps, if any.
        link_deps = []
        try:
            sources = extra_sources + spec.get("sources", [])
        except TypeError:
            print("extra_sources: ", str(extra_sources))
            print('spec.get("sources"): ', str(spec.get("sources")))
            raise
        if sources:
            if self.flavor == "mac" and len(self.archs) > 1:
                # Write subninja file containing compile and link commands scoped to
                # a single arch if a fat binary is being built.
                for arch in self.archs:
                    self.ninja.subninja(self._SubninjaNameForArch(arch))

            pch = None
            if self.flavor == "win":
                gyp.msvs_emulation.VerifyMissingSources(
                    sources, self.abs_build_dir, generator_flags, self.GypPathToNinja
                )
                pch = gyp.msvs_emulation.PrecompiledHeader(
                    self.msvs_settings,
                    config_name,
                    self.GypPathToNinja,
                    self.GypPathToUniqueOutput,
                    self.obj_ext,
                )
            else:
                pch = gyp.xcode_emulation.MacPrefixHeader(
                    self.xcode_settings,
                    self.GypPathToNinja,
                    lambda path, lang: self.GypPathToUniqueOutput(path + "-" + lang),
                )
            link_deps = self.WriteSources(
                self.ninja,
                config_name,
                config,
                sources,
                compile_depends_stamp,
                pch,
                spec,
            )
            # Some actions/rules output 'sources' that are already object files.
            obj_outputs = [f for f in sources if f.endswith(self.obj_ext)]
            if obj_outputs:
                if self.flavor != "mac" or len(self.archs) == 1:
                    link_deps += [self.GypPathToNinja(o) for o in obj_outputs]
                else:
                    print(
                        "Warning: Actions/rules writing object files don't work with "
                        "multiarch targets, dropping. (target %s)" % spec["target_name"]
                    )
        elif self.flavor == "mac" and len(self.archs) > 1:
            link_deps = collections.defaultdict(list)

        compile_deps = self.target.actions_stamp or actions_depends
        if self.flavor == "win" and self.target.type == "static_library":
            self.target.component_objs = link_deps
            self.target.compile_deps = compile_deps

        # Write out a link step, if needed.
        output = None
        is_empty_bundle = not link_deps and not mac_bundle_depends
        if link_deps or self.target.actions_stamp or actions_depends:
            output = self.WriteTarget(
                spec, config_name, config, link_deps, compile_deps
            )
            if self.is_mac_bundle:
                mac_bundle_depends.append(output)

        # Bundle all of the above together, if needed.
        if self.is_mac_bundle:
            output = self.WriteMacBundle(spec, mac_bundle_depends, is_empty_bundle)

        if not output:
            return None

        assert self.target.FinalOutput(), output
        return self.target

    def _WinIdlRule(self, source, prebuild, outputs):
        """Handle the implicit VS .idl rule for one source file. Fills |outputs|
        with files that are generated."""
        outdir, output, vars, flags = self.msvs_settings.GetIdlBuildData(
            source, self.config_name
        )
        outdir = self.GypPathToNinja(outdir)

        def fix_path(path, rel=None):
            path = os.path.join(outdir, path)
            dirname, basename = os.path.split(source)
            root, ext = os.path.splitext(basename)
            path = self.ExpandRuleVariables(path, root, dirname, source, ext, basename)
            if rel:
                path = os.path.relpath(path, rel)
            return path

        vars = [(name, fix_path(value, outdir)) for name, value in vars]
        output = [fix_path(p) for p in output]
        vars.append(("outdir", outdir))
        vars.append(("idlflags", flags))
        input = self.GypPathToNinja(source)
        self.ninja.build(output, "idl", input, variables=vars, order_only=prebuild)
        outputs.extend(output)

    def WriteWinIdlFiles(self, spec, prebuild):
        """Writes rules to match MSVS's implicit idl handling."""
        assert self.flavor == "win"
        if self.msvs_settings.HasExplicitIdlRulesOrActions(spec):
            return []
        outputs = []
        for source in filter(lambda x: x.endswith(".idl"), spec["sources"]):
            self._WinIdlRule(source, prebuild, outputs)
        return outputs

    def WriteActionsRulesCopies(
        self, spec, extra_sources, prebuild, mac_bundle_depends
    ):
        """Write out the Actions, Rules, and Copies steps.  Return a path
        representing the outputs of these steps."""
        outputs = []
        if self.is_mac_bundle:
            mac_bundle_resources = spec.get("mac_bundle_resources", [])[:]
        else:
            mac_bundle_resources = []
        extra_mac_bundle_resources = []

        if "actions" in spec:
            outputs += self.WriteActions(
                spec["actions"], extra_sources, prebuild, extra_mac_bundle_resources
            )
        if "rules" in spec:
            outputs += self.WriteRules(
                spec["rules"],
                extra_sources,
                prebuild,
                mac_bundle_resources,
                extra_mac_bundle_resources,
            )
        if "copies" in spec:
            outputs += self.WriteCopies(spec["copies"], prebuild, mac_bundle_depends)

        if "sources" in spec and self.flavor == "win":
            outputs += self.WriteWinIdlFiles(spec, prebuild)

        if self.xcode_settings and self.xcode_settings.IsIosFramework():
            self.WriteiOSFrameworkHeaders(spec, outputs, prebuild)

        stamp = self.WriteCollapsedDependencies("actions_rules_copies", outputs)

        if self.is_mac_bundle:
            xcassets = self.WriteMacBundleResources(
                extra_mac_bundle_resources + mac_bundle_resources, mac_bundle_depends
            )
            partial_info_plist = self.WriteMacXCassets(xcassets, mac_bundle_depends)
            self.WriteMacInfoPlist(partial_info_plist, mac_bundle_depends)

        return stamp

    def GenerateDescription(self, verb, message, fallback):
        """Generate and return a description of a build step.

        |verb| is the short summary, e.g. ACTION or RULE.
        |message| is a hand-written description, or None if not available.
        |fallback| is the gyp-level name of the step, usable as a fallback.
        """
        if self.toolset != "target":
            verb += "(%s)" % self.toolset
        if message:
            return f"{verb} {self.ExpandSpecial(message)}"
        else:
            return f"{verb} {self.name}: {fallback}"

    def WriteActions(
        self, actions, extra_sources, prebuild, extra_mac_bundle_resources
    ):
        # Actions cd into the base directory.
        env = self.GetToolchainEnv()
        all_outputs = []
        for action in actions:
            # First write out a rule for the action.
            name = "{}_{}".format(action["action_name"], self.hash_for_rules)
            description = self.GenerateDescription(
                "ACTION", action.get("message", None), name
            )
            win_shell_flags = (
                self.msvs_settings.GetRuleShellFlags(action)
                if self.flavor == "win"
                else None
            )
            args = action["action"]
            depfile = action.get("depfile", None)
            if depfile:
                depfile = self.ExpandSpecial(depfile, self.base_to_build)
            pool = "console" if int(action.get("ninja_use_console", 0)) else None
            rule_name, _ = self.WriteNewNinjaRule(
                name, args, description, win_shell_flags, env, pool, depfile=depfile
            )

            inputs = [self.GypPathToNinja(i, env) for i in action["inputs"]]
            if int(action.get("process_outputs_as_sources", False)):
                extra_sources += action["outputs"]
            if int(action.get("process_outputs_as_mac_bundle_resources", False)):
                extra_mac_bundle_resources += action["outputs"]
            outputs = [self.GypPathToNinja(o, env) for o in action["outputs"]]

            # Then write out an edge using the rule.
            self.ninja.build(outputs, rule_name, inputs, order_only=prebuild)
            all_outputs += outputs

            self.ninja.newline()

        return all_outputs

    def WriteRules(
        self,
        rules,
        extra_sources,
        prebuild,
        mac_bundle_resources,
        extra_mac_bundle_resources,
    ):
        env = self.GetToolchainEnv()
        all_outputs = []
        for rule in rules:
            # Skip a rule with no action and no inputs.
            if "action" not in rule and not rule.get("rule_sources", []):
                continue

            # First write out a rule for the rule action.
            name = "{}_{}".format(rule["rule_name"], self.hash_for_rules)

            args = rule["action"]
            description = self.GenerateDescription(
                "RULE",
                rule.get("message", None),
                ("%s " + generator_default_variables["RULE_INPUT_PATH"]) % name,
            )
            win_shell_flags = (
                self.msvs_settings.GetRuleShellFlags(rule)
                if self.flavor == "win"
                else None
            )
            pool = "console" if int(rule.get("ninja_use_console", 0)) else None
            rule_name, args = self.WriteNewNinjaRule(
                name, args, description, win_shell_flags, env, pool
            )

            # TODO: if the command references the outputs directly, we should
            # simplify it to just use $out.

            # Rules can potentially make use of some special variables which
            # must vary per source file.
            # Compute the list of variables we'll need to provide.
            special_locals = ("source", "root", "dirname", "ext", "name")
            needed_variables = {"source"}
            for argument in args:
                for var in special_locals:
                    if "${%s}" % var in argument:
                        needed_variables.add(var)
            needed_variables = sorted(needed_variables)

            def cygwin_munge(path):
                # pylint: disable=cell-var-from-loop
                if win_shell_flags and win_shell_flags.cygwin:
                    return path.replace("\\", "/")
                return path

            inputs = [self.GypPathToNinja(i, env) for i in rule.get("inputs", [])]

            # If there are n source files matching the rule, and m additional rule
            # inputs, then adding 'inputs' to each build edge written below will
            # write m * n inputs. Collapsing reduces this to m + n.
            sources = rule.get("rule_sources", [])
            num_inputs = len(inputs)
            if prebuild:
                num_inputs += 1
            if num_inputs > 2 and len(sources) > 2:
                inputs = [
                    self.WriteCollapsedDependencies(
                        rule["rule_name"], inputs, order_only=prebuild
                    )
                ]
                prebuild = []

            # For each source file, write an edge that generates all the outputs.
            for source in sources:
                source = os.path.normpath(source)
                dirname, basename = os.path.split(source)
                root, ext = os.path.splitext(basename)

                # Gather the list of inputs and outputs, expanding $vars if possible.
                outputs = [
                    self.ExpandRuleVariables(o, root, dirname, source, ext, basename)
                    for o in rule["outputs"]
                ]

                if int(rule.get("process_outputs_as_sources", False)):
                    extra_sources += outputs

                was_mac_bundle_resource = source in mac_bundle_resources
                if was_mac_bundle_resource or int(
                    rule.get("process_outputs_as_mac_bundle_resources", False)
                ):
                    extra_mac_bundle_resources += outputs
                    # Note: This is n_resources * n_outputs_in_rule.
                    # Put to-be-removed items in a set and
                    # remove them all in a single pass
                    # if this becomes a performance issue.
                    if was_mac_bundle_resource:
                        mac_bundle_resources.remove(source)

                extra_bindings = []
                for var in needed_variables:
                    if var == "root":
                        extra_bindings.append(("root", cygwin_munge(root)))
                    elif var == "dirname":
                        # '$dirname' is a parameter to the rule action, which means
                        # it shouldn't be converted to a Ninja path.  But we don't
                        # want $!PRODUCT_DIR in there either.
                        dirname_expanded = self.ExpandSpecial(
                            dirname, self.base_to_build
                        )
                        extra_bindings.append(
                            ("dirname", cygwin_munge(dirname_expanded))
                        )
                    elif var == "source":
                        # '$source' is a parameter to the rule action, which means
                        # it shouldn't be converted to a Ninja path.  But we don't
                        # want $!PRODUCT_DIR in there either.
                        source_expanded = self.ExpandSpecial(source, self.base_to_build)
                        extra_bindings.append(("source", cygwin_munge(source_expanded)))
                    elif var == "ext":
                        extra_bindings.append(("ext", ext))
                    elif var == "name":
                        extra_bindings.append(("name", cygwin_munge(basename)))
                    else:
                        assert var is None, repr(var)

                outputs = [self.GypPathToNinja(o, env) for o in outputs]
                if self.flavor == "win":
                    # WriteNewNinjaRule uses unique_name to create a rsp file on win.
                    extra_bindings.append(
                        ("unique_name", hashlib.md5(outputs[0]).hexdigest())
                    )

                self.ninja.build(
                    outputs,
                    rule_name,
                    self.GypPathToNinja(source),
                    implicit=inputs,
                    order_only=prebuild,
                    variables=extra_bindings,
                )

                all_outputs.extend(outputs)

        return all_outputs

    def WriteCopies(self, copies, prebuild, mac_bundle_depends):
        outputs = []
        if self.xcode_settings:
            extra_env = self.xcode_settings.GetPerTargetSettings()
            env = self.GetToolchainEnv(additional_settings=extra_env)
        else:
            env = self.GetToolchainEnv()
        for to_copy in copies:
            for path in to_copy["files"]:
                # Normalize the path so trailing slashes don't confuse us.
                path = os.path.normpath(path)
                basename = os.path.split(path)[1]
                src = self.GypPathToNinja(path, env)
                dst = self.GypPathToNinja(
                    os.path.join(to_copy["destination"], basename), env
                )
                outputs += self.ninja.build(dst, "copy", src, order_only=prebuild)
                if self.is_mac_bundle:
                    # gyp has mac_bundle_resources to copy things into a bundle's
                    # Resources folder, but there's no built-in way to copy files
                    # to other places in the bundle.
                    # Hence, some targets use copies for this.
                    # Check if this file is copied into the current bundle,
                    # and if so add it to the bundle depends so
                    # that dependent targets get rebuilt if the copy input changes.
                    if dst.startswith(
                        self.xcode_settings.GetBundleContentsFolderPath()
                    ):
                        mac_bundle_depends.append(dst)

        return outputs

    def WriteiOSFrameworkHeaders(self, spec, outputs, prebuild):
        """Prebuild steps to generate hmap files and copy headers to destination."""
        framework = self.ComputeMacBundleOutput()
        all_sources = spec["sources"]
        copy_headers = spec["mac_framework_headers"]
        output = self.GypPathToUniqueOutput("headers.hmap")
        self.xcode_settings.header_map_path = output
        all_headers = map(
            self.GypPathToNinja, filter(lambda x: x.endswith(".h"), all_sources)
        )
        variables = [
            ("framework", framework),
            ("copy_headers", map(self.GypPathToNinja, copy_headers)),
        ]
        outputs.extend(
            self.ninja.build(
                output,
                "compile_ios_framework_headers",
                all_headers,
                variables=variables,
                order_only=prebuild,
            )
        )

    def WriteMacBundleResources(self, resources, bundle_depends):
        """Writes ninja edges for 'mac_bundle_resources'."""
        xcassets = []

        extra_env = self.xcode_settings.GetPerTargetSettings()
        env = self.GetSortedXcodeEnv(additional_settings=extra_env)
        env = self.ComputeExportEnvString(env)
        isBinary = self.xcode_settings.IsBinaryOutputFormat(self.config_name)

        for output, res in gyp.xcode_emulation.GetMacBundleResources(
            generator_default_variables["PRODUCT_DIR"],
            self.xcode_settings,
            map(self.GypPathToNinja, resources),
        ):
            output = self.ExpandSpecial(output)
            if os.path.splitext(output)[-1] != ".xcassets":
                self.ninja.build(
                    output,
                    "mac_tool",
                    res,
                    variables=[
                        ("mactool_cmd", "copy-bundle-resource"),
                        ("env", env),
                        ("binary", isBinary),
                    ],
                )
                bundle_depends.append(output)
            else:
                xcassets.append(res)
        return xcassets

    def WriteMacXCassets(self, xcassets, bundle_depends):
        """Writes ninja edges for 'mac_bundle_resources' .xcassets files.

        This add an invocation of 'actool' via the 'mac_tool.py' helper script.
        It assumes that the assets catalogs define at least one imageset and
        thus an Assets.car file will be generated in the application resources
        directory. If this is not the case, then the build will probably be done
        at each invocation of ninja."""
        if not xcassets:
            return

        extra_arguments = {}
        settings_to_arg = {
            "XCASSETS_APP_ICON": "app-icon",
            "XCASSETS_LAUNCH_IMAGE": "launch-image",
        }
        settings = self.xcode_settings.xcode_settings[self.config_name]
        for settings_key, arg_name in settings_to_arg.items():
            value = settings.get(settings_key)
            if value:
                extra_arguments[arg_name] = value

        partial_info_plist = None
        if extra_arguments:
            partial_info_plist = self.GypPathToUniqueOutput(
                "assetcatalog_generated_info.plist"
            )
            extra_arguments["output-partial-info-plist"] = partial_info_plist

        outputs = []
        outputs.append(
            os.path.join(self.xcode_settings.GetBundleResourceFolder(), "Assets.car")
        )
        if partial_info_plist:
            outputs.append(partial_info_plist)

        keys = QuoteShellArgument(json.dumps(extra_arguments), self.flavor)
        extra_env = self.xcode_settings.GetPerTargetSettings()
        env = self.GetSortedXcodeEnv(additional_settings=extra_env)
        env = self.ComputeExportEnvString(env)

        bundle_depends.extend(
            self.ninja.build(
                outputs,
                "compile_xcassets",
                xcassets,
                variables=[("env", env), ("keys", keys)],
            )
        )
        return partial_info_plist

    def WriteMacInfoPlist(self, partial_info_plist, bundle_depends):
        """Write build rules for bundle Info.plist files."""
        info_plist, out, defines, extra_env = gyp.xcode_emulation.GetMacInfoPlist(
            generator_default_variables["PRODUCT_DIR"],
            self.xcode_settings,
            self.GypPathToNinja,
        )
        if not info_plist:
            return
        out = self.ExpandSpecial(out)
        if defines:
            # Create an intermediate file to store preprocessed results.
            intermediate_plist = self.GypPathToUniqueOutput(
                os.path.basename(info_plist)
            )
            defines = " ".join([Define(d, self.flavor) for d in defines])
            info_plist = self.ninja.build(
                intermediate_plist,
                "preprocess_infoplist",
                info_plist,
                variables=[("defines", defines)],
            )

        env = self.GetSortedXcodeEnv(additional_settings=extra_env)
        env = self.ComputeExportEnvString(env)

        if partial_info_plist:
            intermediate_plist = self.GypPathToUniqueOutput("merged_info.plist")
            info_plist = self.ninja.build(
                intermediate_plist, "merge_infoplist", [partial_info_plist, info_plist]
            )

        keys = self.xcode_settings.GetExtraPlistItems(self.config_name)
        keys = QuoteShellArgument(json.dumps(keys), self.flavor)
        isBinary = self.xcode_settings.IsBinaryOutputFormat(self.config_name)
        self.ninja.build(
            out,
            "copy_infoplist",
            info_plist,
            variables=[("env", env), ("keys", keys), ("binary", isBinary)],
        )
        bundle_depends.append(out)

    def WriteSources(
        self,
        ninja_file,
        config_name,
        config,
        sources,
        predepends,
        precompiled_header,
        spec,
    ):
        """Write build rules to compile all of |sources|."""
        if self.toolset == "host":
            self.ninja.variable("ar", "$ar_host")
            self.ninja.variable("cc", "$cc_host")
            self.ninja.variable("cxx", "$cxx_host")
            self.ninja.variable("ld", "$ld_host")
            self.ninja.variable("ldxx", "$ldxx_host")
            self.ninja.variable("nm", "$nm_host")
            self.ninja.variable("readelf", "$readelf_host")

        if self.flavor != "mac" or len(self.archs) == 1:
            return self.WriteSourcesForArch(
                self.ninja,
                config_name,
                config,
                sources,
                predepends,
                precompiled_header,
                spec,
            )
        else:
            return {
                arch: self.WriteSourcesForArch(
                    self.arch_subninjas[arch],
                    config_name,
                    config,
                    sources,
                    predepends,
                    precompiled_header,
                    spec,
                    arch=arch,
                )
                for arch in self.archs
            }

    def WriteSourcesForArch(
        self,
        ninja_file,
        config_name,
        config,
        sources,
        predepends,
        precompiled_header,
        spec,
        arch=None,
    ):
        """Write build rules to compile all of |sources|."""

        extra_defines = []
        if self.flavor == "mac":
            cflags = self.xcode_settings.GetCflags(config_name, arch=arch)
            cflags_c = self.xcode_settings.GetCflagsC(config_name)
            cflags_cc = self.xcode_settings.GetCflagsCC(config_name)
            cflags_objc = ["$cflags_c"] + self.xcode_settings.GetCflagsObjC(config_name)
            cflags_objcc = ["$cflags_cc"] + self.xcode_settings.GetCflagsObjCC(
                config_name
            )
        elif self.flavor == "win":
            asmflags = self.msvs_settings.GetAsmflags(config_name)
            cflags = self.msvs_settings.GetCflags(config_name)
            cflags_c = self.msvs_settings.GetCflagsC(config_name)
            cflags_cc = self.msvs_settings.GetCflagsCC(config_name)
            extra_defines = self.msvs_settings.GetComputedDefines(config_name)
            # See comment at cc_command for why there's two .pdb files.
            pdbpath_c = pdbpath_cc = self.msvs_settings.GetCompilerPdbName(
                config_name, self.ExpandSpecial
            )
            if not pdbpath_c:
                obj = "obj"
                if self.toolset != "target":
                    obj += "." + self.toolset
                pdbpath = os.path.normpath(os.path.join(obj, self.base_dir, self.name))
                pdbpath_c = pdbpath + ".c.pdb"
                pdbpath_cc = pdbpath + ".cc.pdb"
            self.WriteVariableList(ninja_file, "pdbname_c", [pdbpath_c])
            self.WriteVariableList(ninja_file, "pdbname_cc", [pdbpath_cc])
            self.WriteVariableList(ninja_file, "pchprefix", [self.name])
        else:
            cflags = config.get("cflags", [])
            cflags_c = config.get("cflags_c", [])
            cflags_cc = config.get("cflags_cc", [])

        # Respect environment variables related to build, but target-specific
        # flags can still override them.
        if self.toolset == "target":
            cflags_c = (
                os.environ.get("CPPFLAGS", "").split()
                + os.environ.get("CFLAGS", "").split()
                + cflags_c
            )
            cflags_cc = (
                os.environ.get("CPPFLAGS", "").split()
                + os.environ.get("CXXFLAGS", "").split()
                + cflags_cc
            )
        elif self.toolset == "host":
            cflags_c = (
                os.environ.get("CPPFLAGS_host", "").split()
                + os.environ.get("CFLAGS_host", "").split()
                + cflags_c
            )
            cflags_cc = (
                os.environ.get("CPPFLAGS_host", "").split()
                + os.environ.get("CXXFLAGS_host", "").split()
                + cflags_cc
            )

        defines = config.get("defines", []) + extra_defines
        self.WriteVariableList(
            ninja_file, "defines", [Define(d, self.flavor) for d in defines]
        )
        if self.flavor == "win":
            self.WriteVariableList(
                ninja_file, "asmflags", map(self.ExpandSpecial, asmflags)
            )
            self.WriteVariableList(
                ninja_file,
                "rcflags",
                [
                    QuoteShellArgument(self.ExpandSpecial(f), self.flavor)
                    for f in self.msvs_settings.GetRcflags(
                        config_name, self.GypPathToNinja
                    )
                ],
            )

        include_dirs = config.get("include_dirs", [])

        env = self.GetToolchainEnv()
        if self.flavor == "win":
            include_dirs = self.msvs_settings.AdjustIncludeDirs(
                include_dirs, config_name
            )
        self.WriteVariableList(
            ninja_file,
            "includes",
            [
                QuoteShellArgument("-I" + self.GypPathToNinja(i, env), self.flavor)
                for i in include_dirs
            ],
        )

        if self.flavor == "win":
            midl_include_dirs = config.get("midl_include_dirs", [])
            midl_include_dirs = self.msvs_settings.AdjustMidlIncludeDirs(
                midl_include_dirs, config_name
            )
            self.WriteVariableList(
                ninja_file,
                "midl_includes",
                [
                    QuoteShellArgument("-I" + self.GypPathToNinja(i, env), self.flavor)
                    for i in midl_include_dirs
                ],
            )

        pch_commands = precompiled_header.GetPchBuildCommands(arch)
        if self.flavor == "mac":
            # Most targets use no precompiled headers, so only write these if needed.
            for ext, var in [
                ("c", "cflags_pch_c"),
                ("cc", "cflags_pch_cc"),
                ("m", "cflags_pch_objc"),
                ("mm", "cflags_pch_objcc"),
            ]:
                include = precompiled_header.GetInclude(ext, arch)
                if include:
                    ninja_file.variable(var, include)

        arflags = config.get("arflags", [])

        self.WriteVariableList(ninja_file, "cflags", map(self.ExpandSpecial, cflags))
        self.WriteVariableList(
            ninja_file, "cflags_c", map(self.ExpandSpecial, cflags_c)
        )
        self.WriteVariableList(
            ninja_file, "cflags_cc", map(self.ExpandSpecial, cflags_cc)
        )
        if self.flavor == "mac":
            self.WriteVariableList(
                ninja_file, "cflags_objc", map(self.ExpandSpecial, cflags_objc)
            )
            self.WriteVariableList(
                ninja_file, "cflags_objcc", map(self.ExpandSpecial, cflags_objcc)
            )
        self.WriteVariableList(ninja_file, "arflags", map(self.ExpandSpecial, arflags))
        ninja_file.newline()
        outputs = []
        has_rc_source = False
        for source in sources:
            filename, ext = os.path.splitext(source)
            ext = ext[1:]
            obj_ext = self.obj_ext
            if ext in ("cc", "cpp", "cxx"):
                command = "cxx"
                self.target.uses_cpp = True
            elif ext == "c" or (ext == "S" and self.flavor != "win"):
                command = "cc"
            elif ext == "s" and self.flavor != "win":  # Doesn't generate .o.d files.
                command = "cc_s"
            elif (
                self.flavor == "win"
                and ext in ("asm", "S")
                and not self.msvs_settings.HasExplicitAsmRules(spec)
            ):
                command = "asm"
                # Add the _asm suffix as msvs is capable of handling .cc and
                # .asm files of the same name without collision.
                obj_ext = "_asm.obj"
            elif self.flavor == "mac" and ext == "m":
                command = "objc"
            elif self.flavor == "mac" and ext == "mm":
                command = "objcxx"
                self.target.uses_cpp = True
            elif self.flavor == "win" and ext == "rc":
                command = "rc"
                obj_ext = ".res"
                has_rc_source = True
            else:
                # Ignore unhandled extensions.
                continue
            input = self.GypPathToNinja(source)
            output = self.GypPathToUniqueOutput(filename + obj_ext)
            if arch is not None:
                output = AddArch(output, arch)
            implicit = precompiled_header.GetObjDependencies([input], [output], arch)
            variables = []
            if self.flavor == "win":
                variables, output, implicit = precompiled_header.GetFlagsModifications(
                    input,
                    output,
                    implicit,
                    command,
                    cflags_c,
                    cflags_cc,
                    self.ExpandSpecial,
                )
            ninja_file.build(
                output,
                command,
                input,
                implicit=[gch for _, _, gch in implicit],
                order_only=predepends,
                variables=variables,
            )
            outputs.append(output)

        if has_rc_source:
            resource_include_dirs = config.get("resource_include_dirs", include_dirs)
            self.WriteVariableList(
                ninja_file,
                "resource_includes",
                [
                    QuoteShellArgument("-I" + self.GypPathToNinja(i, env), self.flavor)
                    for i in resource_include_dirs
                ],
            )

        self.WritePchTargets(ninja_file, pch_commands)

        ninja_file.newline()
        return outputs

    def WritePchTargets(self, ninja_file, pch_commands):
        """Writes ninja rules to compile prefix headers."""
        if not pch_commands:
            return

        for gch, lang_flag, lang, input in pch_commands:
            var_name = {
                "c": "cflags_pch_c",
                "cc": "cflags_pch_cc",
                "m": "cflags_pch_objc",
                "mm": "cflags_pch_objcc",
            }[lang]

            map = {
                "c": "cc",
                "cc": "cxx",
                "m": "objc",
                "mm": "objcxx",
            }
            cmd = map.get(lang)
            ninja_file.build(gch, cmd, input, variables=[(var_name, lang_flag)])

    def WriteLink(self, spec, config_name, config, link_deps, compile_deps):
        """Write out a link step. Fills out target.binary. """
        if self.flavor != "mac" or len(self.archs) == 1:
            return self.WriteLinkForArch(
                self.ninja, spec, config_name, config, link_deps, compile_deps
            )
        else:
            output = self.ComputeOutput(spec)
            inputs = [
                self.WriteLinkForArch(
                    self.arch_subninjas[arch],
                    spec,
                    config_name,
                    config,
                    link_deps[arch],
                    compile_deps,
                    arch=arch,
                )
                for arch in self.archs
            ]
            extra_bindings = []
            build_output = output
            if not self.is_mac_bundle:
                self.AppendPostbuildVariable(extra_bindings, spec, output, output)

            # TODO(yyanagisawa): more work needed to fix:
            # https://code.google.com/p/gyp/issues/detail?id=411
            if (
                spec["type"] in ("shared_library", "loadable_module")
                and not self.is_mac_bundle
            ):
                extra_bindings.append(("lib", output))
                self.ninja.build(
                    [output, output + ".TOC"],
                    "solipo",
                    inputs,
                    variables=extra_bindings,
                )
            else:
                self.ninja.build(build_output, "lipo", inputs, variables=extra_bindings)
            return output

    def WriteLinkForArch(
        self, ninja_file, spec, config_name, config, link_deps, compile_deps, arch=None
    ):
        """Write out a link step. Fills out target.binary. """
        command = {
            "executable": "link",
            "loadable_module": "solink_module",
            "shared_library": "solink",
        }[spec["type"]]
        command_suffix = ""

        implicit_deps = set()
        solibs = set()
        order_deps = set()

        if compile_deps:
            # Normally, the compiles of the target already depend on compile_deps,
            # but a shared_library target might have no sources and only link together
            # a few static_library deps, so the link step also needs to depend
            # on compile_deps to make sure actions in the shared_library target
            # get run before the link.
            order_deps.add(compile_deps)

        if "dependencies" in spec:
            # Two kinds of dependencies:
            # - Linkable dependencies (like a .a or a .so): add them to the link line.
            # - Non-linkable dependencies (like a rule that generates a file
            #   and writes a stamp file): add them to implicit_deps
            extra_link_deps = set()
            for dep in spec["dependencies"]:
                target = self.target_outputs.get(dep)
                if not target:
                    continue
                linkable = target.Linkable()
                if linkable:
                    new_deps = []
                    if (
                        self.flavor == "win"
                        and target.component_objs
                        and self.msvs_settings.IsUseLibraryDependencyInputs(config_name)
                    ):
                        new_deps = target.component_objs
                        if target.compile_deps:
                            order_deps.add(target.compile_deps)
                    elif self.flavor == "win" and target.import_lib:
                        new_deps = [target.import_lib]
                    elif target.UsesToc(self.flavor):
                        solibs.add(target.binary)
                        implicit_deps.add(target.binary + ".TOC")
                    else:
                        new_deps = [target.binary]
                    for new_dep in new_deps:
                        if new_dep not in extra_link_deps:
                            extra_link_deps.add(new_dep)
                            link_deps.append(new_dep)

                final_output = target.FinalOutput()
                if not linkable or final_output != target.binary:
                    implicit_deps.add(final_output)

        extra_bindings = []
        if self.target.uses_cpp and self.flavor != "win":
            extra_bindings.append(("ld", "$ldxx"))

        output = self.ComputeOutput(spec, arch)
        if arch is None and not self.is_mac_bundle:
            self.AppendPostbuildVariable(extra_bindings, spec, output, output)

        is_executable = spec["type"] == "executable"
        # The ldflags config key is not used on mac or win. On those platforms
        # linker flags are set via xcode_settings and msvs_settings, respectively.
        if self.toolset == "target":
            env_ldflags = os.environ.get("LDFLAGS", "").split()
        elif self.toolset == "host":
            env_ldflags = os.environ.get("LDFLAGS_host", "").split()

        if self.flavor == "mac":
            ldflags = self.xcode_settings.GetLdflags(
                config_name,
                self.ExpandSpecial(generator_default_variables["PRODUCT_DIR"]),
                self.GypPathToNinja,
                arch,
            )
            ldflags = env_ldflags + ldflags
        elif self.flavor == "win":
            manifest_base_name = self.GypPathToUniqueOutput(
                self.ComputeOutputFileName(spec)
            )
            (
                ldflags,
                intermediate_manifest,
                manifest_files,
            ) = self.msvs_settings.GetLdflags(
                config_name,
                self.GypPathToNinja,
                self.ExpandSpecial,
                manifest_base_name,
                output,
                is_executable,
                self.toplevel_build,
            )
            ldflags = env_ldflags + ldflags
            self.WriteVariableList(ninja_file, "manifests", manifest_files)
            implicit_deps = implicit_deps.union(manifest_files)
            if intermediate_manifest:
                self.WriteVariableList(
                    ninja_file, "intermediatemanifest", [intermediate_manifest]
                )
            command_suffix = _GetWinLinkRuleNameSuffix(
                self.msvs_settings.IsEmbedManifest(config_name)
            )
            def_file = self.msvs_settings.GetDefFile(self.GypPathToNinja)
            if def_file:
                implicit_deps.add(def_file)
        else:
            # Respect environment variables related to build, but target-specific
            # flags can still override them.
            ldflags = env_ldflags + config.get("ldflags", [])
            if is_executable and len(solibs):
                rpath = "lib/"
                if self.toolset != "target":
                    rpath += self.toolset
                    ldflags.append(r"-Wl,-rpath=\$$ORIGIN/%s" % rpath)
                else:
                    ldflags.append("-Wl,-rpath=%s" % self.target_rpath)
                ldflags.append("-Wl,-rpath-link=%s" % rpath)
        self.WriteVariableList(ninja_file, "ldflags", map(self.ExpandSpecial, ldflags))

        library_dirs = config.get("library_dirs", [])
        if self.flavor == "win":
            library_dirs = [
                self.msvs_settings.ConvertVSMacros(library_dir, config_name)
                for library_dir in library_dirs
            ]
            library_dirs = [
                "/LIBPATH:"
                + QuoteShellArgument(self.GypPathToNinja(library_dir), self.flavor)
                for library_dir in library_dirs
            ]
        else:
            library_dirs = [
                QuoteShellArgument("-L" + self.GypPathToNinja(library_dir), self.flavor)
                for library_dir in library_dirs
            ]

        libraries = gyp.common.uniquer(
            map(self.ExpandSpecial, spec.get("libraries", []))
        )
        if self.flavor == "mac":
            libraries = self.xcode_settings.AdjustLibraries(libraries, config_name)
        elif self.flavor == "win":
            libraries = self.msvs_settings.AdjustLibraries(libraries)

        self.WriteVariableList(ninja_file, "libs", library_dirs + libraries)

        linked_binary = output

        if command in ("solink", "solink_module"):
            extra_bindings.append(("soname", os.path.split(output)[1]))
            extra_bindings.append(("lib", gyp.common.EncodePOSIXShellArgument(output)))
            if self.flavor != "win":
                link_file_list = output
                if self.is_mac_bundle:
                    # 'Dependency Framework.framework/Versions/A/Dependency Framework'
                    # -> 'Dependency Framework.framework.rsp'
                    link_file_list = self.xcode_settings.GetWrapperName()
                if arch:
                    link_file_list += "." + arch
                link_file_list += ".rsp"
                # If an rspfile contains spaces, ninja surrounds the filename with
                # quotes around it and then passes it to open(), creating a file with
                # quotes in its name (and when looking for the rsp file, the name
                # makes it through bash which strips the quotes) :-/
                link_file_list = link_file_list.replace(" ", "_")
                extra_bindings.append(
                    (
                        "link_file_list",
                        gyp.common.EncodePOSIXShellArgument(link_file_list),
                    )
                )
            if self.flavor == "win":
                extra_bindings.append(("binary", output))
                if (
                    "/NOENTRY" not in ldflags
                    and not self.msvs_settings.GetNoImportLibrary(config_name)
                ):
                    self.target.import_lib = output + ".lib"
                    extra_bindings.append(
                        ("implibflag", "/IMPLIB:%s" % self.target.import_lib)
                    )
                    pdbname = self.msvs_settings.GetPDBName(
                        config_name, self.ExpandSpecial, output + ".pdb"
                    )
                    output = [output, self.target.import_lib]
                    if pdbname:
                        output.append(pdbname)
            elif not self.is_mac_bundle:
                output = [output, output + ".TOC"]
            else:
                command = command + "_notoc"
        elif self.flavor == "win":
            extra_bindings.append(("binary", output))
            pdbname = self.msvs_settings.GetPDBName(
                config_name, self.ExpandSpecial, output + ".pdb"
            )
            if pdbname:
                output = [output, pdbname]

        if len(solibs):
            extra_bindings.append(
                ("solibs", gyp.common.EncodePOSIXShellList(sorted(solibs)))
            )

        ninja_file.build(
            output,
            command + command_suffix,
            link_deps,
            implicit=sorted(implicit_deps),
            order_only=list(order_deps),
            variables=extra_bindings,
        )
        return linked_binary

    def WriteTarget(self, spec, config_name, config, link_deps, compile_deps):
        extra_link_deps = any(
            self.target_outputs.get(dep).Linkable()
            for dep in spec.get("dependencies", [])
            if dep in self.target_outputs
        )
        if spec["type"] == "none" or (not link_deps and not extra_link_deps):
            # TODO(evan): don't call this function for 'none' target types, as
            # it doesn't do anything, and we fake out a 'binary' with a stamp file.
            self.target.binary = compile_deps
            self.target.type = "none"
        elif spec["type"] == "static_library":
            self.target.binary = self.ComputeOutput(spec)
            if (
                self.flavor not in ("mac", "openbsd", "netbsd", "win")
                and not self.is_standalone_static_library
            ):
                self.ninja.build(
                    self.target.binary, "alink_thin", link_deps, order_only=compile_deps
                )
            else:
                variables = []
                if self.xcode_settings:
                    libtool_flags = self.xcode_settings.GetLibtoolflags(config_name)
                    if libtool_flags:
                        variables.append(("libtool_flags", libtool_flags))
                if self.msvs_settings:
                    libflags = self.msvs_settings.GetLibFlags(
                        config_name, self.GypPathToNinja
                    )
                    variables.append(("libflags", libflags))

                if self.flavor != "mac" or len(self.archs) == 1:
                    self.AppendPostbuildVariable(
                        variables, spec, self.target.binary, self.target.binary
                    )
                    self.ninja.build(
                        self.target.binary,
                        "alink",
                        link_deps,
                        order_only=compile_deps,
                        variables=variables,
                    )
                else:
                    inputs = []
                    for arch in self.archs:
                        output = self.ComputeOutput(spec, arch)
                        self.arch_subninjas[arch].build(
                            output,
                            "alink",
                            link_deps[arch],
                            order_only=compile_deps,
                            variables=variables,
                        )
                        inputs.append(output)
                    # TODO: It's not clear if
                    # libtool_flags should be passed to the alink
                    # call that combines single-arch .a files into a fat .a file.
                    self.AppendPostbuildVariable(
                        variables, spec, self.target.binary, self.target.binary
                    )
                    self.ninja.build(
                        self.target.binary,
                        "alink",
                        inputs,
                        # FIXME: test proving order_only=compile_deps isn't
                        # needed.
                        variables=variables,
                    )
        else:
            self.target.binary = self.WriteLink(
                spec, config_name, config, link_deps, compile_deps
            )
        return self.target.binary

    def WriteMacBundle(self, spec, mac_bundle_depends, is_empty):
        assert self.is_mac_bundle
        package_framework = spec["type"] in ("shared_library", "loadable_module")
        output = self.ComputeMacBundleOutput()
        if is_empty:
            output += ".stamp"
        variables = []
        self.AppendPostbuildVariable(
            variables,
            spec,
            output,
            self.target.binary,
            is_command_start=not package_framework,
        )
        if package_framework and not is_empty:
            if spec["type"] == "shared_library" and self.xcode_settings.isIOS:
                self.ninja.build(
                    output,
                    "package_ios_framework",
                    mac_bundle_depends,
                    variables=variables,
                )
            else:
                variables.append(("version", self.xcode_settings.GetFrameworkVersion()))
                self.ninja.build(
                    output, "package_framework", mac_bundle_depends, variables=variables
                )
        else:
            self.ninja.build(output, "stamp", mac_bundle_depends, variables=variables)
        self.target.bundle = output
        return output

    def GetToolchainEnv(self, additional_settings=None):
        """Returns the variables toolchain would set for build steps."""
        env = self.GetSortedXcodeEnv(additional_settings=additional_settings)
        if self.flavor == "win":
            env = self.GetMsvsToolchainEnv(additional_settings=additional_settings)
        return env

    def GetMsvsToolchainEnv(self, additional_settings=None):
        """Returns the variables Visual Studio would set for build steps."""
        return self.msvs_settings.GetVSMacroEnv(
            "$!PRODUCT_DIR", config=self.config_name
        )

    def GetSortedXcodeEnv(self, additional_settings=None):
        """Returns the variables Xcode would set for build steps."""
        assert self.abs_build_dir
        abs_build_dir = self.abs_build_dir
        return gyp.xcode_emulation.GetSortedXcodeEnv(
            self.xcode_settings,
            abs_build_dir,
            os.path.join(abs_build_dir, self.build_to_base),
            self.config_name,
            additional_settings,
        )

    def GetSortedXcodePostbuildEnv(self):
        """Returns the variables Xcode would set for postbuild steps."""
        postbuild_settings = {}
        # CHROMIUM_STRIP_SAVE_FILE is a chromium-specific hack.
        # TODO(thakis): It would be nice to have some general mechanism instead.
        strip_save_file = self.xcode_settings.GetPerTargetSetting(
            "CHROMIUM_STRIP_SAVE_FILE"
        )
        if strip_save_file:
            postbuild_settings["CHROMIUM_STRIP_SAVE_FILE"] = strip_save_file
        return self.GetSortedXcodeEnv(additional_settings=postbuild_settings)

    def AppendPostbuildVariable(
        self, variables, spec, output, binary, is_command_start=False
    ):
        """Adds a 'postbuild' variable if there is a postbuild for |output|."""
        postbuild = self.GetPostbuildCommand(spec, output, binary, is_command_start)
        if postbuild:
            variables.append(("postbuilds", postbuild))

    def GetPostbuildCommand(self, spec, output, output_binary, is_command_start):
        """Returns a shell command that runs all the postbuilds, and removes
        |output| if any of them fails. If |is_command_start| is False, then the
        returned string will start with ' && '."""
        if not self.xcode_settings or spec["type"] == "none" or not output:
            return ""
        output = QuoteShellArgument(output, self.flavor)
        postbuilds = gyp.xcode_emulation.GetSpecPostbuildCommands(spec, quiet=True)
        if output_binary is not None:
            postbuilds = self.xcode_settings.AddImplicitPostbuilds(
                self.config_name,
                os.path.normpath(os.path.join(self.base_to_build, output)),
                QuoteShellArgument(
                    os.path.normpath(os.path.join(self.base_to_build, output_binary)),
                    self.flavor,
                ),
                postbuilds,
                quiet=True,
            )

        if not postbuilds:
            return ""
        # Postbuilds expect to be run in the gyp file's directory, so insert an
        # implicit postbuild to cd to there.
        postbuilds.insert(
            0, gyp.common.EncodePOSIXShellList(["cd", self.build_to_base])
        )
        env = self.ComputeExportEnvString(self.GetSortedXcodePostbuildEnv())
        # G will be non-null if any postbuild fails. Run all postbuilds in a
        # subshell.
        commands = (
            env
            + " ("
            + " && ".join([ninja_syntax.escape(command) for command in postbuilds])
        )
        command_string = (
            commands
            + "); G=$$?; "
            # Remove the final output if any postbuild failed.
            "((exit $$G) || rm -rf %s) " % output
            + "&& exit $$G)"
        )
        if is_command_start:
            return "(" + command_string + " && "
        else:
            return "$ && (" + command_string

    def ComputeExportEnvString(self, env):
        """Given an environment, returns a string looking like
            'export FOO=foo; export BAR="${FOO} bar;'
        that exports |env| to the shell."""
        export_str = []
        for k, v in env:
            export_str.append(
                "export %s=%s;"
                % (k, ninja_syntax.escape(gyp.common.EncodePOSIXShellArgument(v)))
            )
        return " ".join(export_str)

    def ComputeMacBundleOutput(self):
        """Return the 'output' (full output path) to a bundle output directory."""
        assert self.is_mac_bundle
        path = generator_default_variables["PRODUCT_DIR"]
        return self.ExpandSpecial(
            os.path.join(path, self.xcode_settings.GetWrapperName())
        )

    def ComputeOutputFileName(self, spec, type=None):
        """Compute the filename of the final output for the current target."""
        if not type:
            type = spec["type"]

        default_variables = copy.copy(generator_default_variables)
        CalculateVariables(default_variables, {"flavor": self.flavor})

        # Compute filename prefix: the product prefix, or a default for
        # the product type.
        DEFAULT_PREFIX = {
            "loadable_module": default_variables["SHARED_LIB_PREFIX"],
            "shared_library": default_variables["SHARED_LIB_PREFIX"],
            "static_library": default_variables["STATIC_LIB_PREFIX"],
            "executable": default_variables["EXECUTABLE_PREFIX"],
        }
        prefix = spec.get("product_prefix", DEFAULT_PREFIX.get(type, ""))

        # Compute filename extension: the product extension, or a default
        # for the product type.
        DEFAULT_EXTENSION = {
            "loadable_module": default_variables["SHARED_LIB_SUFFIX"],
            "shared_library": default_variables["SHARED_LIB_SUFFIX"],
            "static_library": default_variables["STATIC_LIB_SUFFIX"],
            "executable": default_variables["EXECUTABLE_SUFFIX"],
        }
        extension = spec.get("product_extension")
        if extension:
            extension = "." + extension
        else:
            extension = DEFAULT_EXTENSION.get(type, "")

        if "product_name" in spec:
            # If we were given an explicit name, use that.
            target = spec["product_name"]
        else:
            # Otherwise, derive a name from the target name.
            target = spec["target_name"]
            if prefix == "lib":
                # Snip out an extra 'lib' from libs if appropriate.
                target = StripPrefix(target, "lib")

        if type in (
            "static_library",
            "loadable_module",
            "shared_library",
            "executable",
        ):
            return f"{prefix}{target}{extension}"
        elif type == "none":
            return "%s.stamp" % target
        else:
            raise Exception("Unhandled output type %s" % type)

    def ComputeOutput(self, spec, arch=None):
        """Compute the path for the final output of the spec."""
        type = spec["type"]

        if self.flavor == "win":
            override = self.msvs_settings.GetOutputName(
                self.config_name, self.ExpandSpecial
            )
            if override:
                return override

        if (
            arch is None
            and self.flavor == "mac"
            and type
            in ("static_library", "executable", "shared_library", "loadable_module")
        ):
            filename = self.xcode_settings.GetExecutablePath()
        else:
            filename = self.ComputeOutputFileName(spec, type)

        if arch is None and "product_dir" in spec:
            path = os.path.join(spec["product_dir"], filename)
            return self.ExpandSpecial(path)

        # Some products go into the output root, libraries go into shared library
        # dir, and everything else goes into the normal place.
        type_in_output_root = ["executable", "loadable_module"]
        if self.flavor == "mac" and self.toolset == "target":
            type_in_output_root += ["shared_library", "static_library"]
        elif self.flavor == "win" and self.toolset == "target":
            type_in_output_root += ["shared_library"]

        if arch is not None:
            # Make sure partial executables don't end up in a bundle or the regular
            # output directory.
            archdir = "arch"
            if self.toolset != "target":
                archdir = os.path.join("arch", "%s" % self.toolset)
            return os.path.join(archdir, AddArch(filename, arch))
        elif type in type_in_output_root or self.is_standalone_static_library:
            return filename
        elif type == "shared_library":
            libdir = "lib"
            if self.toolset != "target":
                libdir = os.path.join("lib", "%s" % self.toolset)
            return os.path.join(libdir, filename)
        else:
            return self.GypPathToUniqueOutput(filename, qualified=False)

    def WriteVariableList(self, ninja_file, var, values):
        assert not isinstance(values, str)
        if values is None:
            values = []
        ninja_file.variable(var, " ".join(values))

    def WriteNewNinjaRule(
        self, name, args, description, win_shell_flags, env, pool, depfile=None
    ):
        """Write out a new ninja "rule" statement for a given command.

        Returns the name of the new rule, and a copy of |args| with variables
        expanded."""

        if self.flavor == "win":
            args = [
                self.msvs_settings.ConvertVSMacros(
                    arg, self.base_to_build, config=self.config_name
                )
                for arg in args
            ]
            description = self.msvs_settings.ConvertVSMacros(
                description, config=self.config_name
            )
        elif self.flavor == "mac":
            # |env| is an empty list on non-mac.
            args = [gyp.xcode_emulation.ExpandEnvVars(arg, env) for arg in args]
            description = gyp.xcode_emulation.ExpandEnvVars(description, env)

        # TODO: we shouldn't need to qualify names; we do it because
        # currently the ninja rule namespace is global, but it really
        # should be scoped to the subninja.
        rule_name = self.name
        if self.toolset == "target":
            rule_name += "." + self.toolset
        rule_name += "." + name
        rule_name = re.sub("[^a-zA-Z0-9_]", "_", rule_name)

        # Remove variable references, but not if they refer to the magic rule
        # variables.  This is not quite right, as it also protects these for
        # actions, not just for rules where they are valid. Good enough.
        protect = ["${root}", "${dirname}", "${source}", "${ext}", "${name}"]
        protect = "(?!" + "|".join(map(re.escape, protect)) + ")"
        description = re.sub(protect + r"\$", "_", description)

        # gyp dictates that commands are run from the base directory.
        # cd into the directory before running, and adjust paths in
        # the arguments to point to the proper locations.
        rspfile = None
        rspfile_content = None
        args = [self.ExpandSpecial(arg, self.base_to_build) for arg in args]
        if self.flavor == "win":
            rspfile = rule_name + ".$unique_name.rsp"
            # The cygwin case handles this inside the bash sub-shell.
            run_in = "" if win_shell_flags.cygwin else " " + self.build_to_base
            if win_shell_flags.cygwin:
                rspfile_content = self.msvs_settings.BuildCygwinBashCommandLine(
                    args, self.build_to_base
                )
            else:
                rspfile_content = gyp.msvs_emulation.EncodeRspFileList(
                    args, win_shell_flags.quote)
            command = (
                "%s gyp-win-tool action-wrapper $arch " % sys.executable
                + rspfile
                + run_in
            )
        else:
            env = self.ComputeExportEnvString(env)
            command = gyp.common.EncodePOSIXShellList(args)
            command = "cd %s; " % self.build_to_base + env + command

        # GYP rules/actions express being no-ops by not touching their outputs.
        # Avoid executing downstream dependencies in this case by specifying
        # restat=1 to ninja.
        self.ninja.rule(
            rule_name,
            command,
            description,
            depfile=depfile,
            restat=True,
            pool=pool,
            rspfile=rspfile,
            rspfile_content=rspfile_content,
        )
        self.ninja.newline()

        return rule_name, args


def CalculateVariables(default_variables, params):
    """Calculate additional variables for use in the build (called by gyp)."""
    global generator_additional_non_configuration_keys
    global generator_additional_path_sections
    flavor = gyp.common.GetFlavor(params)
    if flavor == "mac":
        default_variables.setdefault("OS", "mac")
        default_variables.setdefault("SHARED_LIB_SUFFIX", ".dylib")
        default_variables.setdefault(
            "SHARED_LIB_DIR", generator_default_variables["PRODUCT_DIR"]
        )
        default_variables.setdefault(
            "LIB_DIR", generator_default_variables["PRODUCT_DIR"]
        )

        # Copy additional generator configuration data from Xcode, which is shared
        # by the Mac Ninja generator.
        import gyp.generator.xcode as xcode_generator

        generator_additional_non_configuration_keys = getattr(
            xcode_generator, "generator_additional_non_configuration_keys", []
        )
        generator_additional_path_sections = getattr(
            xcode_generator, "generator_additional_path_sections", []
        )
        global generator_extra_sources_for_rules
        generator_extra_sources_for_rules = getattr(
            xcode_generator, "generator_extra_sources_for_rules", []
        )
    elif flavor == "win":
        exts = gyp.MSVSUtil.TARGET_TYPE_EXT
        default_variables.setdefault("OS", "win")
        default_variables["EXECUTABLE_SUFFIX"] = "." + exts["executable"]
        default_variables["STATIC_LIB_PREFIX"] = ""
        default_variables["STATIC_LIB_SUFFIX"] = "." + exts["static_library"]
        default_variables["SHARED_LIB_PREFIX"] = ""
        default_variables["SHARED_LIB_SUFFIX"] = "." + exts["shared_library"]

        # Copy additional generator configuration data from VS, which is shared
        # by the Windows Ninja generator.
        import gyp.generator.msvs as msvs_generator

        generator_additional_non_configuration_keys = getattr(
            msvs_generator, "generator_additional_non_configuration_keys", []
        )
        generator_additional_path_sections = getattr(
            msvs_generator, "generator_additional_path_sections", []
        )

        gyp.msvs_emulation.CalculateCommonVariables(default_variables, params)
    else:
        operating_system = flavor
        if flavor == "android":
            operating_system = "linux"  # Keep this legacy behavior for now.
        default_variables.setdefault("OS", operating_system)
        default_variables.setdefault("SHARED_LIB_SUFFIX", ".so")
        default_variables.setdefault(
            "SHARED_LIB_DIR", os.path.join("$!PRODUCT_DIR", "lib")
        )
        default_variables.setdefault("LIB_DIR", os.path.join("$!PRODUCT_DIR", "obj"))


def ComputeOutputDir(params):
    """Returns the path from the toplevel_dir to the build output directory."""
    # generator_dir: relative path from pwd to where make puts build files.
    # Makes migrating from make to ninja easier, ninja doesn't put anything here.
    generator_dir = os.path.relpath(params["options"].generator_output or ".")

    # output_dir: relative path from generator_dir to the build directory.
    output_dir = params.get("generator_flags", {}).get("output_dir", "out")

    # Relative path from source root to our output files.  e.g. "out"
    return os.path.normpath(os.path.join(generator_dir, output_dir))


def CalculateGeneratorInputInfo(params):
    """Called by __init__ to initialize generator values based on params."""
    # E.g. "out/gypfiles"
    toplevel = params["options"].toplevel_dir
    qualified_out_dir = os.path.normpath(
        os.path.join(toplevel, ComputeOutputDir(params), "gypfiles")
    )

    global generator_filelist_paths
    generator_filelist_paths = {
        "toplevel": toplevel,
        "qualified_out_dir": qualified_out_dir,
    }


def OpenOutput(path, mode="w"):
    """Open |path| for writing, creating directories if necessary."""
    gyp.common.EnsureDirExists(path)
    return open(path, mode)


def CommandWithWrapper(cmd, wrappers, prog):
    wrapper = wrappers.get(cmd, "")
    if wrapper:
        return wrapper + " " + prog
    return prog


def GetDefaultConcurrentLinks():
    """Returns a best-guess for a number of concurrent links."""
    pool_size = int(os.environ.get("GYP_LINK_CONCURRENCY", 0))
    if pool_size:
        return pool_size

    if sys.platform in ("win32", "cygwin"):
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

        # VS 2015 uses 20% more working set than VS 2013 and can consume all RAM
        # on a 64 GB machine.
        mem_limit = max(1, stat.ullTotalPhys // (5 * (2 ** 30)))  # total / 5GB
        hard_cap = max(1, int(os.environ.get("GYP_LINK_CONCURRENCY_MAX", 2 ** 32)))
        return min(mem_limit, hard_cap)
    elif sys.platform.startswith("linux"):
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as meminfo:
                memtotal_re = re.compile(r"^MemTotal:\s*(\d*)\s*kB")
                for line in meminfo:
                    match = memtotal_re.match(line)
                    if not match:
                        continue
                    # Allow 8Gb per link on Linux because Gold is quite memory hungry
                    return max(1, int(match.group(1)) // (8 * (2 ** 20)))
        return 1
    elif sys.platform == "darwin":
        try:
            avail_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
            # A static library debug build of Chromium's unit_tests takes ~2.7GB, so
            # 4GB per ld process allows for some more bloat.
            return max(1, avail_bytes // (4 * (2 ** 30)))  # total / 4GB
        except subprocess.CalledProcessError:
            return 1
    else:
        # TODO(scottmg): Implement this for other platforms.
        return 1


def _GetWinLinkRuleNameSuffix(embed_manifest):
    """Returns the suffix used to select an appropriate linking rule depending on
    whether the manifest embedding is enabled."""
    return "_embed" if embed_manifest else ""


def _AddWinLinkRules(master_ninja, embed_manifest):
    """Adds link rules for Windows platform to |master_ninja|."""

    def FullLinkCommand(ldcmd, out, binary_type):
        resource_name = {"exe": "1", "dll": "2"}[binary_type]
        return (
            "%(python)s gyp-win-tool link-with-manifests $arch %(embed)s "
            '%(out)s "%(ldcmd)s" %(resname)s $mt $rc "$intermediatemanifest" '
            "$manifests"
            % {
                "python": sys.executable,
                "out": out,
                "ldcmd": ldcmd,
                "resname": resource_name,
                "embed": embed_manifest,
            }
        )

    rule_name_suffix = _GetWinLinkRuleNameSuffix(embed_manifest)
    use_separate_mspdbsrv = int(os.environ.get("GYP_USE_SEPARATE_MSPDBSRV", "0")) != 0
    dlldesc = "LINK%s(DLL) $binary" % rule_name_suffix.upper()
    dllcmd = (
        "%s gyp-win-tool link-wrapper $arch %s "
        "$ld /nologo $implibflag /DLL /OUT:$binary "
        "@$binary.rsp" % (sys.executable, use_separate_mspdbsrv)
    )
    dllcmd = FullLinkCommand(dllcmd, "$binary", "dll")
    master_ninja.rule(
        "solink" + rule_name_suffix,
        description=dlldesc,
        command=dllcmd,
        rspfile="$binary.rsp",
        rspfile_content="$libs $in_newline $ldflags",
        restat=True,
        pool="link_pool",
    )
    master_ninja.rule(
        "solink_module" + rule_name_suffix,
        description=dlldesc,
        command=dllcmd,
        rspfile="$binary.rsp",
        rspfile_content="$libs $in_newline $ldflags",
        restat=True,
        pool="link_pool",
    )
    # Note that ldflags goes at the end so that it has the option of
    # overriding default settings earlier in the command line.
    exe_cmd = (
        "%s gyp-win-tool link-wrapper $arch %s "
        "$ld /nologo /OUT:$binary @$binary.rsp"
        % (sys.executable, use_separate_mspdbsrv)
    )
    exe_cmd = FullLinkCommand(exe_cmd, "$binary", "exe")
    master_ninja.rule(
        "link" + rule_name_suffix,
        description="LINK%s $binary" % rule_name_suffix.upper(),
        command=exe_cmd,
        rspfile="$binary.rsp",
        rspfile_content="$in_newline $libs $ldflags",
        pool="link_pool",
    )


def GenerateOutputForConfig(target_list, target_dicts, data, params, config_name):
    options = params["options"]
    flavor = gyp.common.GetFlavor(params)
    generator_flags = params.get("generator_flags", {})

    # build_dir: relative path from source root to our output files.
    # e.g. "out/Debug"
    build_dir = os.path.normpath(os.path.join(ComputeOutputDir(params), config_name))

    toplevel_build = os.path.join(options.toplevel_dir, build_dir)

    master_ninja_file = OpenOutput(os.path.join(toplevel_build, "build.ninja"))
    master_ninja = ninja_syntax.Writer(master_ninja_file, width=120)

    # Put build-time support tools in out/{config_name}.
    gyp.common.CopyTool(flavor, toplevel_build, generator_flags)

    # Grab make settings for CC/CXX.
    # The rules are
    # - The priority from low to high is gcc/g++, the 'make_global_settings' in
    #   gyp, the environment variable.
    # - If there is no 'make_global_settings' for CC.host/CXX.host or
    #   'CC_host'/'CXX_host' environment variable, cc_host/cxx_host should be set
    #   to cc/cxx.
    if flavor == "win":
        ar = "lib.exe"
        # cc and cxx must be set to the correct architecture by overriding with one
        # of cl_x86 or cl_x64 below.
        cc = "UNSET"
        cxx = "UNSET"
        ld = "link.exe"
        ld_host = "$ld"
    else:
        ar = "ar"
        cc = "cc"
        cxx = "c++"
        ld = "$cc"
        ldxx = "$cxx"
        ld_host = "$cc_host"
        ldxx_host = "$cxx_host"

    ar_host = ar
    cc_host = None
    cxx_host = None
    cc_host_global_setting = None
    cxx_host_global_setting = None
    clang_cl = None
    nm = "nm"
    nm_host = "nm"
    readelf = "readelf"
    readelf_host = "readelf"

    build_file, _, _ = gyp.common.ParseQualifiedTarget(target_list[0])
    make_global_settings = data[build_file].get("make_global_settings", [])
    build_to_root = gyp.common.InvertRelativePath(build_dir, options.toplevel_dir)
    wrappers = {}
    for key, value in make_global_settings:
        if key == "AR":
            ar = os.path.join(build_to_root, value)
        if key == "AR.host":
            ar_host = os.path.join(build_to_root, value)
        if key == "CC":
            cc = os.path.join(build_to_root, value)
            if cc.endswith("clang-cl"):
                clang_cl = cc
        if key == "CXX":
            cxx = os.path.join(build_to_root, value)
        if key == "CC.host":
            cc_host = os.path.join(build_to_root, value)
            cc_host_global_setting = value
        if key == "CXX.host":
            cxx_host = os.path.join(build_to_root, value)
            cxx_host_global_setting = value
        if key == "LD":
            ld = os.path.join(build_to_root, value)
        if key == "LD.host":
            ld_host = os.path.join(build_to_root, value)
        if key == "LDXX":
            ldxx = os.path.join(build_to_root, value)
        if key == "LDXX.host":
            ldxx_host = os.path.join(build_to_root, value)
        if key == "NM":
            nm = os.path.join(build_to_root, value)
        if key == "NM.host":
            nm_host = os.path.join(build_to_root, value)
        if key == "READELF":
            readelf = os.path.join(build_to_root, value)
        if key == "READELF.host":
            readelf_host = os.path.join(build_to_root, value)
        if key.endswith("_wrapper"):
            wrappers[key[: -len("_wrapper")]] = os.path.join(build_to_root, value)

    # Support wrappers from environment variables too.
    for key, value in os.environ.items():
        if key.lower().endswith("_wrapper"):
            key_prefix = key[: -len("_wrapper")]
            key_prefix = re.sub(r"\.HOST$", ".host", key_prefix)
            wrappers[key_prefix] = os.path.join(build_to_root, value)

    mac_toolchain_dir = generator_flags.get("mac_toolchain_dir", None)
    if mac_toolchain_dir:
        wrappers["LINK"] = "export DEVELOPER_DIR='%s' &&" % mac_toolchain_dir

    if flavor == "win":
        configs = [
            target_dicts[qualified_target]["configurations"][config_name]
            for qualified_target in target_list
        ]
        shared_system_includes = None
        if not generator_flags.get("ninja_use_custom_environment_files", 0):
            shared_system_includes = gyp.msvs_emulation.ExtractSharedMSVSSystemIncludes(
                configs, generator_flags
            )
        cl_paths = gyp.msvs_emulation.GenerateEnvironmentFiles(
            toplevel_build, generator_flags, shared_system_includes, OpenOutput
        )
        for arch, path in sorted(cl_paths.items()):
            if clang_cl:
                # If we have selected clang-cl, use that instead.
                path = clang_cl
            command = CommandWithWrapper(
                "CC", wrappers, QuoteShellArgument(path, "win")
            )
            if clang_cl:
                # Use clang-cl to cross-compile for x86 or x86_64.
                command += " -m32" if arch == "x86" else " -m64"
            master_ninja.variable("cl_" + arch, command)

    cc = GetEnvironFallback(["CC_target", "CC"], cc)
    master_ninja.variable("cc", CommandWithWrapper("CC", wrappers, cc))
    cxx = GetEnvironFallback(["CXX_target", "CXX"], cxx)
    master_ninja.variable("cxx", CommandWithWrapper("CXX", wrappers, cxx))

    if flavor == "win":
        master_ninja.variable("ld", ld)
        master_ninja.variable("idl", "midl.exe")
        master_ninja.variable("ar", ar)
        master_ninja.variable("rc", "rc.exe")
        master_ninja.variable("ml_x86", "ml.exe")
        master_ninja.variable("ml_x64", "ml64.exe")
        master_ninja.variable("mt", "mt.exe")
    else:
        master_ninja.variable("ld", CommandWithWrapper("LINK", wrappers, ld))
        master_ninja.variable("ldxx", CommandWithWrapper("LINK", wrappers, ldxx))
        master_ninja.variable("ar", GetEnvironFallback(["AR_target", "AR"], ar))
        if flavor != "mac":
            # Mac does not use readelf/nm for .TOC generation, so avoiding polluting
            # the master ninja with extra unused variables.
            master_ninja.variable("nm", GetEnvironFallback(["NM_target", "NM"], nm))
            master_ninja.variable(
                "readelf", GetEnvironFallback(["READELF_target", "READELF"], readelf)
            )

    if generator_supports_multiple_toolsets:
        if not cc_host:
            cc_host = cc
        if not cxx_host:
            cxx_host = cxx

        master_ninja.variable("ar_host", GetEnvironFallback(["AR_host"], ar_host))
        master_ninja.variable("nm_host", GetEnvironFallback(["NM_host"], nm_host))
        master_ninja.variable(
            "readelf_host", GetEnvironFallback(["READELF_host"], readelf_host)
        )
        cc_host = GetEnvironFallback(["CC_host"], cc_host)
        cxx_host = GetEnvironFallback(["CXX_host"], cxx_host)

        # The environment variable could be used in 'make_global_settings', like
        # ['CC.host', '$(CC)'] or ['CXX.host', '$(CXX)'], transform them here.
        if "$(CC)" in cc_host and cc_host_global_setting:
            cc_host = cc_host_global_setting.replace("$(CC)", cc)
        if "$(CXX)" in cxx_host and cxx_host_global_setting:
            cxx_host = cxx_host_global_setting.replace("$(CXX)", cxx)
        master_ninja.variable(
            "cc_host", CommandWithWrapper("CC.host", wrappers, cc_host)
        )
        master_ninja.variable(
            "cxx_host", CommandWithWrapper("CXX.host", wrappers, cxx_host)
        )
        if flavor == "win":
            master_ninja.variable("ld_host", ld_host)
        else:
            master_ninja.variable(
                "ld_host", CommandWithWrapper("LINK", wrappers, ld_host)
            )
            master_ninja.variable(
                "ldxx_host", CommandWithWrapper("LINK", wrappers, ldxx_host)
            )

    master_ninja.newline()

    master_ninja.pool("link_pool", depth=GetDefaultConcurrentLinks())
    master_ninja.newline()

    deps = "msvc" if flavor == "win" else "gcc"

    if flavor != "win":
        master_ninja.rule(
            "cc",
            description="CC $out",
            command=(
                "$cc -MMD -MF $out.d $defines $includes $cflags $cflags_c "
                "$cflags_pch_c -c $in -o $out"
            ),
            depfile="$out.d",
            deps=deps,
        )
        master_ninja.rule(
            "cc_s",
            description="CC $out",
            command=(
                "$cc $defines $includes $cflags $cflags_c "
                "$cflags_pch_c -c $in -o $out"
            ),
        )
        master_ninja.rule(
            "cxx",
            description="CXX $out",
            command=(
                "$cxx -MMD -MF $out.d $defines $includes $cflags $cflags_cc "
                "$cflags_pch_cc -c $in -o $out"
            ),
            depfile="$out.d",
            deps=deps,
        )
    else:
        # TODO(scottmg) Separate pdb names is a test to see if it works around
        # http://crbug.com/142362. It seems there's a race between the creation of
        # the .pdb by the precompiled header step for .cc and the compilation of
        # .c files. This should be handled by mspdbsrv, but rarely errors out with
        #   c1xx : fatal error C1033: cannot open program database
        # By making the rules target separate pdb files this might be avoided.
        cc_command = (
            "ninja -t msvc -e $arch " + "-- "
            "$cc /nologo /showIncludes /FC "
            "@$out.rsp /c $in /Fo$out /Fd$pdbname_c "
        )
        cxx_command = (
            "ninja -t msvc -e $arch " + "-- "
            "$cxx /nologo /showIncludes /FC "
            "@$out.rsp /c $in /Fo$out /Fd$pdbname_cc "
        )
        master_ninja.rule(
            "cc",
            description="CC $out",
            command=cc_command,
            rspfile="$out.rsp",
            rspfile_content="$defines $includes $cflags $cflags_c",
            deps=deps,
        )
        master_ninja.rule(
            "cxx",
            description="CXX $out",
            command=cxx_command,
            rspfile="$out.rsp",
            rspfile_content="$defines $includes $cflags $cflags_cc",
            deps=deps,
        )
        master_ninja.rule(
            "idl",
            description="IDL $in",
            command=(
                "%s gyp-win-tool midl-wrapper $arch $outdir "
                "$tlb $h $dlldata $iid $proxy $in "
                "$midl_includes $idlflags" % sys.executable
            ),
        )
        master_ninja.rule(
            "rc",
            description="RC $in",
            # Note: $in must be last otherwise rc.exe complains.
            command=(
                "%s gyp-win-tool rc-wrapper "
                "$arch $rc $defines $resource_includes $rcflags /fo$out $in"
                % sys.executable
            ),
        )
        master_ninja.rule(
            "asm",
            description="ASM $out",
            command=(
                "%s gyp-win-tool asm-wrapper "
                "$arch $asm $defines $includes $asmflags /c /Fo $out $in"
                % sys.executable
            ),
        )

    if flavor != "mac" and flavor != "win":
        master_ninja.rule(
            "alink",
            description="AR $out",
            command="rm -f $out && $ar rcs $arflags $out $in",
        )
        master_ninja.rule(
            "alink_thin",
            description="AR $out",
            command="rm -f $out && $ar rcsT $arflags $out $in",
        )

        # This allows targets that only need to depend on $lib's API to declare an
        # order-only dependency on $lib.TOC and avoid relinking such downstream
        # dependencies when $lib changes only in non-public ways.
        # The resulting string leaves an uninterpolated %{suffix} which
        # is used in the final substitution below.
        mtime_preserving_solink_base = (
            "if [ ! -e $lib -o ! -e $lib.TOC ]; then "
            "%(solink)s && %(extract_toc)s > $lib.TOC; else "
            "%(solink)s && %(extract_toc)s > $lib.tmp && "
            "if ! cmp -s $lib.tmp $lib.TOC; then mv $lib.tmp $lib.TOC ; "
            "fi; fi"
            % {
                "solink": "$ld -shared $ldflags -o $lib -Wl,-soname=$soname %(suffix)s",
                "extract_toc": (
                    "{ $readelf -d $lib | grep SONAME ; "
                    "$nm -gD -f p $lib | cut -f1-2 -d' '; }"
                ),
            }
        )

        master_ninja.rule(
            "solink",
            description="SOLINK $lib",
            restat=True,
            command=mtime_preserving_solink_base
            % {"suffix": "@$link_file_list"},  # noqa: E501
            rspfile="$link_file_list",
            rspfile_content=(
                "-Wl,--whole-archive $in $solibs -Wl," "--no-whole-archive $libs"
            ),
            pool="link_pool",
        )
        master_ninja.rule(
            "solink_module",
            description="SOLINK(module) $lib",
            restat=True,
            command=mtime_preserving_solink_base % {"suffix": "@$link_file_list"},
            rspfile="$link_file_list",
            rspfile_content="-Wl,--start-group $in $solibs $libs -Wl,--end-group",
            pool="link_pool",
        )
        master_ninja.rule(
            "link",
            description="LINK $out",
            command=(
                "$ld $ldflags -o $out "
                "-Wl,--start-group $in $solibs $libs -Wl,--end-group"
            ),
            pool="link_pool",
        )
    elif flavor == "win":
        master_ninja.rule(
            "alink",
            description="LIB $out",
            command=(
                "%s gyp-win-tool link-wrapper $arch False "
                "$ar /nologo /ignore:4221 /OUT:$out @$out.rsp" % sys.executable
            ),
            rspfile="$out.rsp",
            rspfile_content="$in_newline $libflags",
        )
        _AddWinLinkRules(master_ninja, embed_manifest=True)
        _AddWinLinkRules(master_ninja, embed_manifest=False)
    else:
        master_ninja.rule(
            "objc",
            description="OBJC $out",
            command=(
                "$cc -MMD -MF $out.d $defines $includes $cflags $cflags_objc "
                "$cflags_pch_objc -c $in -o $out"
            ),
            depfile="$out.d",
            deps=deps,
        )
        master_ninja.rule(
            "objcxx",
            description="OBJCXX $out",
            command=(
                "$cxx -MMD -MF $out.d $defines $includes $cflags $cflags_objcc "
                "$cflags_pch_objcc -c $in -o $out"
            ),
            depfile="$out.d",
            deps=deps,
        )
        master_ninja.rule(
            "alink",
            description="LIBTOOL-STATIC $out, POSTBUILDS",
            command="rm -f $out && "
            "./gyp-mac-tool filter-libtool libtool $libtool_flags "
            "-static -o $out $in"
            "$postbuilds",
        )
        master_ninja.rule(
            "lipo",
            description="LIPO $out, POSTBUILDS",
            command="rm -f $out && lipo -create $in -output $out$postbuilds",
        )
        master_ninja.rule(
            "solipo",
            description="SOLIPO $out, POSTBUILDS",
            command=(
                "rm -f $lib $lib.TOC && lipo -create $in -output $lib$postbuilds &&"
                "%(extract_toc)s > $lib.TOC"
                % {
                    "extract_toc": "{ otool -l $lib | grep LC_ID_DYLIB -A 5; "
                    "nm -gP $lib | cut -f1-2 -d' ' | grep -v U$$; true; }"
                }
            ),
        )

        # Record the public interface of $lib in $lib.TOC. See the corresponding
        # comment in the posix section above for details.
        solink_base = "$ld %(type)s $ldflags -o $lib %(suffix)s"
        mtime_preserving_solink_base = (
            "if [ ! -e $lib -o ! -e $lib.TOC ] || "
            # Always force dependent targets to relink if this library
            # reexports something. Handling this correctly would require
            # recursive TOC dumping but this is rare in practice, so punt.
            "otool -l $lib | grep -q LC_REEXPORT_DYLIB ; then "
            "%(solink)s && %(extract_toc)s > $lib.TOC; "
            "else "
            "%(solink)s && %(extract_toc)s > $lib.tmp && "
            "if ! cmp -s $lib.tmp $lib.TOC; then "
            "mv $lib.tmp $lib.TOC ; "
            "fi; "
            "fi"
            % {
                "solink": solink_base,
                "extract_toc": "{ otool -l $lib | grep LC_ID_DYLIB -A 5; "
                "nm -gP $lib | cut -f1-2 -d' ' | grep -v U$$; true; }",
            }
        )

        solink_suffix = "@$link_file_list$postbuilds"
        master_ninja.rule(
            "solink",
            description="SOLINK $lib, POSTBUILDS",
            restat=True,
            command=mtime_preserving_solink_base
            % {"suffix": solink_suffix, "type": "-shared"},
            rspfile="$link_file_list",
            rspfile_content="$in $solibs $libs",
            pool="link_pool",
        )
        master_ninja.rule(
            "solink_notoc",
            description="SOLINK $lib, POSTBUILDS",
            restat=True,
            command=solink_base % {"suffix": solink_suffix, "type": "-shared"},
            rspfile="$link_file_list",
            rspfile_content="$in $solibs $libs",
            pool="link_pool",
        )

        master_ninja.rule(
            "solink_module",
            description="SOLINK(module) $lib, POSTBUILDS",
            restat=True,
            command=mtime_preserving_solink_base
            % {"suffix": solink_suffix, "type": "-bundle"},
            rspfile="$link_file_list",
            rspfile_content="$in $solibs $libs",
            pool="link_pool",
        )
        master_ninja.rule(
            "solink_module_notoc",
            description="SOLINK(module) $lib, POSTBUILDS",
            restat=True,
            command=solink_base % {"suffix": solink_suffix, "type": "-bundle"},
            rspfile="$link_file_list",
            rspfile_content="$in $solibs $libs",
            pool="link_pool",
        )

        master_ninja.rule(
            "link",
            description="LINK $out, POSTBUILDS",
            command=("$ld $ldflags -o $out " "$in $solibs $libs$postbuilds"),
            pool="link_pool",
        )
        master_ninja.rule(
            "preprocess_infoplist",
            description="PREPROCESS INFOPLIST $out",
            command=(
                "$cc -E -P -Wno-trigraphs -x c $defines $in -o $out && "
                "plutil -convert xml1 $out $out"
            ),
        )
        master_ninja.rule(
            "copy_infoplist",
            description="COPY INFOPLIST $in",
            command="$env ./gyp-mac-tool copy-info-plist $in $out $binary $keys",
        )
        master_ninja.rule(
            "merge_infoplist",
            description="MERGE INFOPLISTS $in",
            command="$env ./gyp-mac-tool merge-info-plist $out $in",
        )
        master_ninja.rule(
            "compile_xcassets",
            description="COMPILE XCASSETS $in",
            command="$env ./gyp-mac-tool compile-xcassets $keys $in",
        )
        master_ninja.rule(
            "compile_ios_framework_headers",
            description="COMPILE HEADER MAPS AND COPY FRAMEWORK HEADERS $in",
            command="$env ./gyp-mac-tool compile-ios-framework-header-map $out "
            "$framework $in && $env ./gyp-mac-tool "
            "copy-ios-framework-headers $framework $copy_headers",
        )
        master_ninja.rule(
            "mac_tool",
            description="MACTOOL $mactool_cmd $in",
            command="$env ./gyp-mac-tool $mactool_cmd $in $out $binary",
        )
        master_ninja.rule(
            "package_framework",
            description="PACKAGE FRAMEWORK $out, POSTBUILDS",
            command="./gyp-mac-tool package-framework $out $version$postbuilds "
            "&& touch $out",
        )
        master_ninja.rule(
            "package_ios_framework",
            description="PACKAGE IOS FRAMEWORK $out, POSTBUILDS",
            command="./gyp-mac-tool package-ios-framework $out $postbuilds "
            "&& touch $out",
        )
    if flavor == "win":
        master_ninja.rule(
            "stamp",
            description="STAMP $out",
            command="%s gyp-win-tool stamp $out" % sys.executable,
        )
    else:
        master_ninja.rule(
            "stamp", description="STAMP $out", command="${postbuilds}touch $out"
        )
    if flavor == "win":
        master_ninja.rule(
            "copy",
            description="COPY $in $out",
            command="%s gyp-win-tool recursive-mirror $in $out" % sys.executable,
        )
    elif flavor == "zos":
        master_ninja.rule(
            "copy",
            description="COPY $in $out",
            command="rm -rf $out && cp -fRP $in $out",
        )
    else:
        master_ninja.rule(
            "copy",
            description="COPY $in $out",
            command="ln -f $in $out 2>/dev/null || (rm -rf $out && cp -af $in $out)",
        )
    master_ninja.newline()

    all_targets = set()
    for build_file in params["build_files"]:
        for target in gyp.common.AllTargets(
            target_list, target_dicts, os.path.normpath(build_file)
        ):
            all_targets.add(target)
    all_outputs = set()

    # target_outputs is a map from qualified target name to a Target object.
    target_outputs = {}
    # target_short_names is a map from target short name to a list of Target
    # objects.
    target_short_names = {}

    # short name of targets that were skipped because they didn't contain anything
    # interesting.
    # NOTE: there may be overlap between this an non_empty_target_names.
    empty_target_names = set()

    # Set of non-empty short target names.
    # NOTE: there may be overlap between this an empty_target_names.
    non_empty_target_names = set()

    for qualified_target in target_list:
        # qualified_target is like: third_party/icu/icu.gyp:icui18n#target
        build_file, name, toolset = gyp.common.ParseQualifiedTarget(qualified_target)

        this_make_global_settings = data[build_file].get("make_global_settings", [])
        assert make_global_settings == this_make_global_settings, (
            "make_global_settings needs to be the same for all targets. "
            f"{this_make_global_settings} vs. {make_global_settings}"
        )

        spec = target_dicts[qualified_target]
        if flavor == "mac":
            gyp.xcode_emulation.MergeGlobalXcodeSettingsToSpec(data[build_file], spec)

        # If build_file is a symlink, we must not follow it because there's a chance
        # it could point to a path above toplevel_dir, and we cannot correctly deal
        # with that case at the moment.
        build_file = gyp.common.RelativePath(build_file, options.toplevel_dir, False)

        qualified_target_for_hash = gyp.common.QualifiedTarget(
            build_file, name, toolset
        )
        qualified_target_for_hash = qualified_target_for_hash.encode("utf-8")
        hash_for_rules = hashlib.md5(qualified_target_for_hash).hexdigest()

        base_path = os.path.dirname(build_file)
        obj = "obj"
        if toolset != "target":
            obj += "." + toolset
        output_file = os.path.join(obj, base_path, name + ".ninja")

        ninja_output = StringIO()
        writer = NinjaWriter(
            hash_for_rules,
            target_outputs,
            base_path,
            build_dir,
            ninja_output,
            toplevel_build,
            output_file,
            flavor,
            toplevel_dir=options.toplevel_dir,
        )

        target = writer.WriteSpec(spec, config_name, generator_flags)

        if ninja_output.tell() > 0:
            # Only create files for ninja files that actually have contents.
            with OpenOutput(os.path.join(toplevel_build, output_file)) as ninja_file:
                ninja_file.write(ninja_output.getvalue())
            ninja_output.close()
            master_ninja.subninja(output_file)

        if target:
            if name != target.FinalOutput() and spec["toolset"] == "target":
                target_short_names.setdefault(name, []).append(target)
            target_outputs[qualified_target] = target
            if qualified_target in all_targets:
                all_outputs.add(target.FinalOutput())
            non_empty_target_names.add(name)
        else:
            empty_target_names.add(name)

    if target_short_names:
        # Write a short name to build this target.  This benefits both the
        # "build chrome" case as well as the gyp tests, which expect to be
        # able to run actions and build libraries by their short name.
        master_ninja.newline()
        master_ninja.comment("Short names for targets.")
        for short_name in sorted(target_short_names):
            master_ninja.build(
                short_name,
                "phony",
                [x.FinalOutput() for x in target_short_names[short_name]],
            )

    # Write phony targets for any empty targets that weren't written yet. As
    # short names are  not necessarily unique only do this for short names that
    # haven't already been output for another target.
    empty_target_names = empty_target_names - non_empty_target_names
    if empty_target_names:
        master_ninja.newline()
        master_ninja.comment("Empty targets (output for completeness).")
        for name in sorted(empty_target_names):
            master_ninja.build(name, "phony")

    if all_outputs:
        master_ninja.newline()
        master_ninja.build("all", "phony", sorted(all_outputs))
        master_ninja.default(generator_flags.get("default_target", "all"))

    master_ninja_file.close()


def PerformBuild(data, configurations, params):
    options = params["options"]
    for config in configurations:
        builddir = os.path.join(options.toplevel_dir, "out", config)
        arguments = ["ninja", "-C", builddir]
        print(f"Building [{config}]: {arguments}")
        subprocess.check_call(arguments)


def CallGenerateOutputForConfig(arglist):
    # Ignore the interrupt signal so that the parent process catches it and
    # kills all multiprocessing children.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    (target_list, target_dicts, data, params, config_name) = arglist
    GenerateOutputForConfig(target_list, target_dicts, data, params, config_name)


def GenerateOutput(target_list, target_dicts, data, params):
    # Update target_dicts for iOS device builds.
    target_dicts = gyp.xcode_emulation.CloneConfigurationForDeviceAndEmulator(
        target_dicts
    )

    user_config = params.get("generator_flags", {}).get("config", None)
    if gyp.common.GetFlavor(params) == "win":
        target_list, target_dicts = MSVSUtil.ShardTargets(target_list, target_dicts)
        target_list, target_dicts = MSVSUtil.InsertLargePdbShims(
            target_list, target_dicts, generator_default_variables
        )

    if user_config:
        GenerateOutputForConfig(target_list, target_dicts, data, params, user_config)
    else:
        config_names = target_dicts[target_list[0]]["configurations"]
        if params["parallel"]:
            try:
                pool = multiprocessing.Pool(len(config_names))
                arglists = []
                for config_name in config_names:
                    arglists.append(
                        (target_list, target_dicts, data, params, config_name)
                    )
                pool.map(CallGenerateOutputForConfig, arglists)
            except KeyboardInterrupt as e:
                pool.terminate()
                raise e
        else:
            for config_name in config_names:
                GenerateOutputForConfig(
                    target_list, target_dicts, data, params, config_name
                )# Copyright (c) 2013 Google Inc. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

# Notes:
#
# This is all roughly based on the Makefile system used by the Linux
# kernel, but is a non-recursive make -- we put the entire dependency
# graph in front of make and let it figure it out.
#
# The code below generates a separate .mk file for each target, but
# all are sourced by the top-level Makefile.  This means that all
# variables in .mk-files clobber one another.  Be careful to use :=
# where appropriate for immediate evaluation, and similarly to watch
# that you're not relying on a variable value to last between different
# .mk files.
#
# TODOs:
#
# Global settings and utility functions are currently stuffed in the
# toplevel Makefile.  It may make sense to generate some .mk files on
# the side to keep the files readable.


import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback

import hashlib

generator_default_variables = {
    "EXECUTABLE_PREFIX": "",
    "EXECUTABLE_SUFFIX": "",
    "STATIC_LIB_PREFIX": "lib",
    "SHARED_LIB_PREFIX": "lib",
    "STATIC_LIB_SUFFIX": ".a",
    "INTERMEDIATE_DIR": "$(obj).$(TOOLSET)/$(TARGET)/geni",
    "SHARED_INTERMEDIATE_DIR": "$(obj)/gen",
    "PRODUCT_DIR": "$(builddir)",
    "RULE_INPUT_ROOT": "%(INPUT_ROOT)s",  # This gets expanded by Python.
    "RULE_INPUT_DIRNAME": "%(INPUT_DIRNAME)s",  # This gets expanded by Python.
    "RULE_INPUT_PATH": "$(abspath $<)",
    "RULE_INPUT_EXT": "$(suffix $<)",
    "RULE_INPUT_NAME": "$(notdir $<)",
    "CONFIGURATION_NAME": "$(BUILDTYPE)",
}

# Make supports multiple toolsets
generator_supports_multiple_toolsets = True

# Request sorted dependencies in the order from dependents to dependencies.
generator_wants_sorted_dependencies = False

# Placates pylint.
generator_additional_non_configuration_keys = []
generator_additional_path_sections = []
generator_extra_sources_for_rules = []
generator_filelist_paths = None


def CalculateVariables(default_variables, params):
    """Calculate additional variables for use in the build (called by gyp)."""
    flavor = gyp.common.GetFlavor(params)
    if flavor == "mac":
        default_variables.setdefault("OS", "mac")
        default_variables.setdefault("SHARED_LIB_SUFFIX", ".dylib")
        default_variables.setdefault(
            "SHARED_LIB_DIR", generator_default_variables["PRODUCT_DIR"]
        )
        default_variables.setdefault(
            "LIB_DIR", generator_default_variables["PRODUCT_DIR"]
        )

        # Copy additional generator configuration data from Xcode, which is shared
        # by the Mac Make generator.
        import gyp.generator.xcode as xcode_generator

        global generator_additional_non_configuration_keys
        generator_additional_non_configuration_keys = getattr(
            xcode_generator, "generator_additional_non_configuration_keys", []
        )
        global generator_additional_path_sections
        generator_additional_path_sections = getattr(
            xcode_generator, "generator_additional_path_sections", []
        )
        global generator_extra_sources_for_rules
        generator_extra_sources_for_rules = getattr(
            xcode_generator, "generator_extra_sources_for_rules", []
        )
        COMPILABLE_EXTENSIONS.update({".m": "objc", ".mm": "objcxx"})
    else:
        operating_system = flavor
        if flavor == "android":
            operating_system = "linux"  # Keep this legacy behavior for now.
        default_variables.setdefault("OS", operating_system)
        if flavor == "aix":
            default_variables.setdefault("SHARED_LIB_SUFFIX", ".a")
        else:
            default_variables.setdefault("SHARED_LIB_SUFFIX", ".so")
        default_variables.setdefault("SHARED_LIB_DIR", "$(builddir)/lib.$(TOOLSET)")
        default_variables.setdefault("LIB_DIR", "$(obj).$(TOOLSET)")


def CalculateGeneratorInputInfo(params):
    """Calculate the generator specific info that gets fed to input (called by
    gyp)."""
    generator_flags = params.get("generator_flags", {})
    android_ndk_version = generator_flags.get("android_ndk_version", None)
    # Android NDK requires a strict link order.
    if android_ndk_version:
        global generator_wants_sorted_dependencies
        generator_wants_sorted_dependencies = True

    output_dir = params["options"].generator_output or params["options"].toplevel_dir
    builddir_name = generator_flags.get("output_dir", "out")
    qualified_out_dir = os.path.normpath(
        os.path.join(output_dir, builddir_name, "gypfiles")
    )

    global generator_filelist_paths
    generator_filelist_paths = {
        "toplevel": params["options"].toplevel_dir,
        "qualified_out_dir": qualified_out_dir,
    }


# The .d checking code below uses these functions:
# wildcard, sort, foreach, shell, wordlist
# wildcard can handle spaces, the rest can't.
# Since I could find no way to make foreach work with spaces in filenames
# correctly, the .d files have spaces replaced with another character. The .d
# file for
#     Chromium\ Framework.framework/foo
# is for example
#     out/Release/.deps/out/Release/Chromium?Framework.framework/foo
# This is the replacement character.
SPACE_REPLACEMENT = "?"


LINK_COMMANDS_LINUX = """\
quiet_cmd_alink = AR($(TOOLSET)) $@
cmd_alink = rm -f $@ && $(AR.$(TOOLSET)) crs $@ $(filter %.o,$^)

quiet_cmd_alink_thin = AR($(TOOLSET)) $@
cmd_alink_thin = rm -f $@ && $(AR.$(TOOLSET)) crsT $@ $(filter %.o,$^)

# Due to circular dependencies between libraries :(, we wrap the
# special "figure out circular dependencies" flags around the entire
# input list during linking.
quiet_cmd_link = LINK($(TOOLSET)) $@
cmd_link = $(LINK.$(TOOLSET)) -o $@ $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -Wl,--start-group $(LD_INPUTS) $(LIBS) -Wl,--end-group

# We support two kinds of shared objects (.so):
# 1) shared_library, which is just bundling together many dependent libraries
# into a link line.
# 2) loadable_module, which is generating a module intended for dlopen().
#
# They differ only slightly:
# In the former case, we want to package all dependent code into the .so.
# In the latter case, we want to package just the API exposed by the
# outermost module.
# This means shared_library uses --whole-archive, while loadable_module doesn't.
# (Note that --whole-archive is incompatible with the --start-group used in
# normal linking.)

# Other shared-object link notes:
# - Set SONAME to the library filename so our binaries don't reference
# the local, absolute paths used on the link command-line.
quiet_cmd_solink = SOLINK($(TOOLSET)) $@
cmd_solink = $(LINK.$(TOOLSET)) -o $@ -shared $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -Wl,-soname=$(@F) -Wl,--whole-archive $(LD_INPUTS) -Wl,--no-whole-archive $(LIBS)

quiet_cmd_solink_module = SOLINK_MODULE($(TOOLSET)) $@
cmd_solink_module = $(LINK.$(TOOLSET)) -o $@ -shared $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -Wl,-soname=$(@F) -Wl,--start-group $(filter-out FORCE_DO_CMD, $^) -Wl,--end-group $(LIBS)
"""  # noqa: E501

LINK_COMMANDS_MAC = """\
quiet_cmd_alink = LIBTOOL-STATIC $@
cmd_alink = rm -f $@ && ./gyp-mac-tool filter-libtool libtool $(GYP_LIBTOOLFLAGS) -static -o $@ $(filter %.o,$^)

quiet_cmd_link = LINK($(TOOLSET)) $@
cmd_link = $(LINK.$(TOOLSET)) $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o "$@" $(LD_INPUTS) $(LIBS)

quiet_cmd_solink = SOLINK($(TOOLSET)) $@
cmd_solink = $(LINK.$(TOOLSET)) -shared $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o "$@" $(LD_INPUTS) $(LIBS)

quiet_cmd_solink_module = SOLINK_MODULE($(TOOLSET)) $@
cmd_solink_module = $(LINK.$(TOOLSET)) -bundle $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o $@ $(filter-out FORCE_DO_CMD, $^) $(LIBS)
"""  # noqa: E501

LINK_COMMANDS_ANDROID = """\
quiet_cmd_alink = AR($(TOOLSET)) $@
cmd_alink = rm -f $@ && $(AR.$(TOOLSET)) crs $@ $(filter %.o,$^)

quiet_cmd_alink_thin = AR($(TOOLSET)) $@
cmd_alink_thin = rm -f $@ && $(AR.$(TOOLSET)) crsT $@ $(filter %.o,$^)

# Due to circular dependencies between libraries :(, we wrap the
# special "figure out circular dependencies" flags around the entire
# input list during linking.
quiet_cmd_link = LINK($(TOOLSET)) $@
quiet_cmd_link_host = LINK($(TOOLSET)) $@
cmd_link = $(LINK.$(TOOLSET)) $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o $@ -Wl,--start-group $(LD_INPUTS) -Wl,--end-group $(LIBS)
cmd_link_host = $(LINK.$(TOOLSET)) $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o $@ -Wl,--start-group $(LD_INPUTS) -Wl,--end-group $(LIBS)

# Other shared-object link notes:
# - Set SONAME to the library filename so our binaries don't reference
# the local, absolute paths used on the link command-line.
quiet_cmd_solink = SOLINK($(TOOLSET)) $@
cmd_solink = $(LINK.$(TOOLSET)) -shared $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -Wl,-soname=$(@F) -o $@ -Wl,--whole-archive $(LD_INPUTS) -Wl,--no-whole-archive $(LIBS)

quiet_cmd_solink_module = SOLINK_MODULE($(TOOLSET)) $@
cmd_solink_module = $(LINK.$(TOOLSET)) -shared $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -Wl,-soname=$(@F) -o $@ -Wl,--start-group $(filter-out FORCE_DO_CMD, $^) -Wl,--end-group $(LIBS)
quiet_cmd_solink_module_host = SOLINK_MODULE($(TOOLSET)) $@
cmd_solink_module_host = $(LINK.$(TOOLSET)) -shared $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -Wl,-soname=$(@F) -o $@ $(filter-out FORCE_DO_CMD, $^) $(LIBS)
"""  # noqa: E501


LINK_COMMANDS_AIX = """\
quiet_cmd_alink = AR($(TOOLSET)) $@
cmd_alink = rm -f $@ && $(AR.$(TOOLSET)) -X32_64 crs $@ $(filter %.o,$^)

quiet_cmd_alink_thin = AR($(TOOLSET)) $@
cmd_alink_thin = rm -f $@ && $(AR.$(TOOLSET)) -X32_64 crs $@ $(filter %.o,$^)

quiet_cmd_link = LINK($(TOOLSET)) $@
cmd_link = $(LINK.$(TOOLSET)) $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o $@ $(LD_INPUTS) $(LIBS)

quiet_cmd_solink = SOLINK($(TOOLSET)) $@
cmd_solink = $(LINK.$(TOOLSET)) -shared $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o $@ $(LD_INPUTS) $(LIBS)

quiet_cmd_solink_module = SOLINK_MODULE($(TOOLSET)) $@
cmd_solink_module = $(LINK.$(TOOLSET)) -shared $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o $@ $(filter-out FORCE_DO_CMD, $^) $(LIBS)
"""  # noqa: E501


LINK_COMMANDS_OS390 = """\
quiet_cmd_alink = AR($(TOOLSET)) $@
cmd_alink = rm -f $@ && $(AR.$(TOOLSET)) crs $@ $(filter %.o,$^)

quiet_cmd_alink_thin = AR($(TOOLSET)) $@
cmd_alink_thin = rm -f $@ && $(AR.$(TOOLSET)) crsT $@ $(filter %.o,$^)

quiet_cmd_link = LINK($(TOOLSET)) $@
cmd_link = $(LINK.$(TOOLSET)) $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o $@ $(LD_INPUTS) $(LIBS)

quiet_cmd_solink = SOLINK($(TOOLSET)) $@
cmd_solink = $(LINK.$(TOOLSET)) $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o $@ $(LD_INPUTS) $(LIBS) -Wl,DLL

quiet_cmd_solink_module = SOLINK_MODULE($(TOOLSET)) $@
cmd_solink_module = $(LINK.$(TOOLSET)) $(GYP_LDFLAGS) $(LDFLAGS.$(TOOLSET)) -o $@ $(filter-out FORCE_DO_CMD, $^) $(LIBS) -Wl,DLL
"""  # noqa: E501


# Header of toplevel Makefile.
# This should go into the build tree, but it's easier to keep it here for now.
SHARED_HEADER = (
    """\
# We borrow heavily from the kernel build setup, though we are simpler since
# we don't have Kconfig tweaking settings on us.

# The implicit make rules have it looking for RCS files, among other things.
# We instead explicitly write all the rules we care about.
# It's even quicker (saves ~200ms) to pass -r on the command line.
MAKEFLAGS=-r

# The source directory tree.
srcdir := %(srcdir)s
abs_srcdir := $(abspath $(srcdir))

# The name of the builddir.
builddir_name ?= %(builddir)s

# The V=1 flag on command line makes us verbosely print command lines.
ifdef V
  quiet=
else
  quiet=quiet_
endif

# Specify BUILDTYPE=Release on the command line for a release build.
BUILDTYPE ?= %(default_configuration)s

# Directory all our build output goes into.
# Note that this must be two directories beneath src/ for unit tests to pass,
# as they reach into the src/ directory for data with relative paths.
builddir ?= $(builddir_name)/$(BUILDTYPE)
abs_builddir := $(abspath $(builddir))
depsdir := $(builddir)/.deps

# Object output directory.
obj := $(builddir)/obj
abs_obj := $(abspath $(obj))

# We build up a list of every single one of the targets so we can slurp in the
# generated dependency rule Makefiles in one pass.
all_deps :=

%(make_global_settings)s

CC.target ?= %(CC.target)s
CFLAGS.target ?= $(CPPFLAGS) $(CFLAGS)
CXX.target ?= %(CXX.target)s
CXXFLAGS.target ?= $(CPPFLAGS) $(CXXFLAGS)
LINK.target ?= %(LINK.target)s
LDFLAGS.target ?= $(LDFLAGS)
AR.target ?= $(AR)

# C++ apps need to be linked with g++.
LINK ?= $(CXX.target)

# TODO(evan): move all cross-compilation logic to gyp-time so we don't need
# to replicate this environment fallback in make as well.
CC.host ?= %(CC.host)s
CFLAGS.host ?= $(CPPFLAGS_host) $(CFLAGS_host)
CXX.host ?= %(CXX.host)s
CXXFLAGS.host ?= $(CPPFLAGS_host) $(CXXFLAGS_host)
LINK.host ?= %(LINK.host)s
LDFLAGS.host ?= $(LDFLAGS_host)
AR.host ?= %(AR.host)s

# Define a dir function that can handle spaces.
# http://www.gnu.org/software/make/manual/make.html#Syntax-of-Functions
# "leading spaces cannot appear in the text of the first argument as written.
# These characters can be put into the argument value by variable substitution."
empty :=
space := $(empty) $(empty)

# http://stackoverflow.com/questions/1189781/using-make-dir-or-notdir-on-a-path-with-spaces
replace_spaces = $(subst $(space),"""
    + SPACE_REPLACEMENT
    + """,$1)
unreplace_spaces = $(subst """
    + SPACE_REPLACEMENT
    + """,$(space),$1)
dirx = $(call unreplace_spaces,$(dir $(call replace_spaces,$1)))

# Flags to make gcc output dependency info.  Note that you need to be
# careful here to use the flags that ccache and distcc can understand.
# We write to a dep file on the side first and then rename at the end
# so we can't end up with a broken dep file.
depfile = $(depsdir)/$(call replace_spaces,$@).d
DEPFLAGS = %(makedep_args)s -MF $(depfile).raw

# We have to fixup the deps output in a few ways.
# (1) the file output should mention the proper .o file.
# ccache or distcc lose the path to the target, so we convert a rule of
# the form:
#   foobar.o: DEP1 DEP2
# into
#   path/to/foobar.o: DEP1 DEP2
# (2) we want missing files not to cause us to fail to build.
# We want to rewrite
#   foobar.o: DEP1 DEP2 \\
#               DEP3
# to
#   DEP1:
#   DEP2:
#   DEP3:
# so if the files are missing, they're just considered phony rules.
# We have to do some pretty insane escaping to get those backslashes
# and dollar signs past make, the shell, and sed at the same time.
# Doesn't work with spaces, but that's fine: .d files have spaces in
# their names replaced with other characters."""
    r"""
define fixup_dep
# The depfile may not exist if the input file didn't have any #includes.
touch $(depfile).raw
# Fixup path as in (1).
sed -e "s|^$(notdir $@)|$@|" $(depfile).raw >> $(depfile)
# Add extra rules as in (2).
# We remove slashes and replace spaces with new lines;
# remove blank lines;
# delete the first line and append a colon to the remaining lines.
sed -e 's|\\||' -e 'y| |\n|' $(depfile).raw |\
  grep -v '^$$'                             |\
  sed -e 1d -e 's|$$|:|'                     \
    >> $(depfile)
rm $(depfile).raw
endef
"""
    """
# Command definitions:
# - cmd_foo is the actual command to run;
# - quiet_cmd_foo is the brief-output summary of the command.

quiet_cmd_cc = CC($(TOOLSET)) $@
cmd_cc = $(CC.$(TOOLSET)) -o $@ $< $(GYP_CFLAGS) $(DEPFLAGS) $(CFLAGS.$(TOOLSET)) -c

quiet_cmd_cxx = CXX($(TOOLSET)) $@
cmd_cxx = $(CXX.$(TOOLSET)) -o $@ $< $(GYP_CXXFLAGS) $(DEPFLAGS) $(CXXFLAGS.$(TOOLSET)) -c
%(extra_commands)s
quiet_cmd_touch = TOUCH $@
cmd_touch = touch $@

quiet_cmd_copy = COPY $@
# send stderr to /dev/null to ignore messages when linking directories.
cmd_copy = ln -f "$<" "$@" 2>/dev/null || (rm -rf "$@" && cp %(copy_archive_args)s "$<" "$@")

%(link_commands)s
"""  # noqa: E501
    r"""
# Define an escape_quotes function to escape single quotes.
# This allows us to handle quotes properly as long as we always use
# use single quotes and escape_quotes.
escape_quotes = $(subst ','\'',$(1))
# This comment is here just to include a ' to unconfuse syntax highlighting.
# Define an escape_vars function to escape '$' variable syntax.
# This allows us to read/write command lines with shell variables (e.g.
# $LD_LIBRARY_PATH), without triggering make substitution.
escape_vars = $(subst $$,$$$$,$(1))
# Helper that expands to a shell command to echo a string exactly as it is in
# make. This uses printf instead of echo because printf's behaviour with respect
# to escape sequences is more portable than echo's across different shells
# (e.g., dash, bash).
exact_echo = printf '%%s\n' '$(call escape_quotes,$(1))'
"""
    """
# Helper to compare the command we're about to run against the command
# we logged the last time we ran the command.  Produces an empty
# string (false) when the commands match.
# Tricky point: Make has no string-equality test function.
# The kernel uses the following, but it seems like it would have false
# positives, where one string reordered its arguments.
#   arg_check = $(strip $(filter-out $(cmd_$(1)), $(cmd_$@)) \\
#                       $(filter-out $(cmd_$@), $(cmd_$(1))))
# We instead substitute each for the empty string into the other, and
# say they're equal if both substitutions produce the empty string.
# .d files contain """
    + SPACE_REPLACEMENT
    + """ instead of spaces, take that into account.
command_changed = $(or $(subst $(cmd_$(1)),,$(cmd_$(call replace_spaces,$@))),\\
                       $(subst $(cmd_$(call replace_spaces,$@)),,$(cmd_$(1))))

# Helper that is non-empty when a prerequisite changes.
# Normally make does this implicitly, but we force rules to always run
# so we can check their command lines.
#   $? -- new prerequisites
#   $| -- order-only dependencies
prereq_changed = $(filter-out FORCE_DO_CMD,$(filter-out $|,$?))

# Helper that executes all postbuilds until one fails.
define do_postbuilds
  @E=0;\\
  for p in $(POSTBUILDS); do\\
    eval $$p;\\
    E=$$?;\\
    if [ $$E -ne 0 ]; then\\
      break;\\
    fi;\\
  done;\\
  if [ $$E -ne 0 ]; then\\
    rm -rf "$@";\\
    exit $$E;\\
  fi
endef

# do_cmd: run a command via the above cmd_foo names, if necessary.
# Should always run for a given target to handle command-line changes.
# Second argument, if non-zero, makes it do asm/C/C++ dependency munging.
# Third argument, if non-zero, makes it do POSTBUILDS processing.
# Note: We intentionally do NOT call dirx for depfile, since it contains """
    + SPACE_REPLACEMENT
    + """ for
# spaces already and dirx strips the """
    + SPACE_REPLACEMENT
    + """ characters.
define do_cmd
$(if $(or $(command_changed),$(prereq_changed)),
  @$(call exact_echo,  $($(quiet)cmd_$(1)))
  @mkdir -p "$(call dirx,$@)" "$(dir $(depfile))"
  $(if $(findstring flock,$(word %(flock_index)d,$(cmd_$1))),
    @$(cmd_$(1))
    @echo "  $(quiet_cmd_$(1)): Finished",
    @$(cmd_$(1))
  )
  @$(call exact_echo,$(call escape_vars,cmd_$(call replace_spaces,$@) := $(cmd_$(1)))) > $(depfile)
  @$(if $(2),$(fixup_dep))
  $(if $(and $(3), $(POSTBUILDS)),
    $(call do_postbuilds)
  )
)
endef

# Declare the "%(default_target)s" target first so it is the default,
# even though we don't have the deps yet.
.PHONY: %(default_target)s
%(default_target)s:

# make looks for ways to re-generate included makefiles, but in our case, we
# don't have a direct way. Explicitly telling make that it has nothing to do
# for them makes it go faster.
%%.d: ;

# Use FORCE_DO_CMD to force a target to run.  Should be coupled with
# do_cmd.
.PHONY: FORCE_DO_CMD
FORCE_DO_CMD:

"""  # noqa: E501
)

SHARED_HEADER_MAC_COMMANDS = """
quiet_cmd_objc = CXX($(TOOLSET)) $@
cmd_objc = $(CC.$(TOOLSET)) $(GYP_OBJCFLAGS) $(DEPFLAGS) -c -o $@ $<

quiet_cmd_objcxx = CXX($(TOOLSET)) $@
cmd_objcxx = $(CXX.$(TOOLSET)) $(GYP_OBJCXXFLAGS) $(DEPFLAGS) -c -o $@ $<

# Commands for precompiled header files.
quiet_cmd_pch_c = CXX($(TOOLSET)) $@
cmd_pch_c = $(CC.$(TOOLSET)) $(GYP_PCH_CFLAGS) $(DEPFLAGS) $(CXXFLAGS.$(TOOLSET)) -c -o $@ $<
quiet_cmd_pch_cc = CXX($(TOOLSET)) $@
cmd_pch_cc = $(CC.$(TOOLSET)) $(GYP_PCH_CXXFLAGS) $(DEPFLAGS) $(CXXFLAGS.$(TOOLSET)) -c -o $@ $<
quiet_cmd_pch_m = CXX($(TOOLSET)) $@
cmd_pch_m = $(CC.$(TOOLSET)) $(GYP_PCH_OBJCFLAGS) $(DEPFLAGS) -c -o $@ $<
quiet_cmd_pch_mm = CXX($(TOOLSET)) $@
cmd_pch_mm = $(CC.$(TOOLSET)) $(GYP_PCH_OBJCXXFLAGS) $(DEPFLAGS) -c -o $@ $<

# gyp-mac-tool is written next to the root Makefile by gyp.
# Use $(4) for the command, since $(2) and $(3) are used as flag by do_cmd
# already.
quiet_cmd_mac_tool = MACTOOL $(4) $<
cmd_mac_tool = ./gyp-mac-tool $(4) $< "$@"

quiet_cmd_mac_package_framework = PACKAGE FRAMEWORK $@
cmd_mac_package_framework = ./gyp-mac-tool package-framework "$@" $(4)

quiet_cmd_infoplist = INFOPLIST $@
cmd_infoplist = $(CC.$(TOOLSET)) -E -P -Wno-trigraphs -x c $(INFOPLIST_DEFINES) "$<" -o "$@"
"""  # noqa: E501


def WriteRootHeaderSuffixRules(writer):
    extensions = sorted(COMPILABLE_EXTENSIONS.keys(), key=str.lower)

    writer.write("# Suffix rules, putting all outputs into $(obj).\n")
    for ext in extensions:
        writer.write("$(obj).$(TOOLSET)/%%.o: $(srcdir)/%%%s FORCE_DO_CMD\n" % ext)
        writer.write("\t@$(call do_cmd,%s,1)\n" % COMPILABLE_EXTENSIONS[ext])

    writer.write("\n# Try building from generated source, too.\n")
    for ext in extensions:
        writer.write(
            "$(obj).$(TOOLSET)/%%.o: $(obj).$(TOOLSET)/%%%s FORCE_DO_CMD\n" % ext
        )
        writer.write("\t@$(call do_cmd,%s,1)\n" % COMPILABLE_EXTENSIONS[ext])
    writer.write("\n")
    for ext in extensions:
        writer.write("$(obj).$(TOOLSET)/%%.o: $(obj)/%%%s FORCE_DO_CMD\n" % ext)
        writer.write("\t@$(call do_cmd,%s,1)\n" % COMPILABLE_EXTENSIONS[ext])
    writer.write("\n")


SHARED_HEADER_SUFFIX_RULES_COMMENT1 = """\
# Suffix rules, putting all outputs into $(obj).
"""


SHARED_HEADER_SUFFIX_RULES_COMMENT2 = """\
# Try building from generated source, too.
"""


SHARED_FOOTER = """\
# "all" is a concatenation of the "all" targets from all the included
# sub-makefiles. This is just here to clarify.
all:

# Add in dependency-tracking rules.  $(all_deps) is the list of every single
# target in our tree. Only consider the ones with .d (dependency) info:
d_files := $(wildcard $(foreach f,$(all_deps),$(depsdir)/$(f).d))
ifneq ($(d_files),)
  include $(d_files)
endif
"""

header = """\
# This file is generated by gyp; do not edit.

"""

# Maps every compilable file extension to the do_cmd that compiles it.
COMPILABLE_EXTENSIONS = {
    ".c": "cc",
    ".cc": "cxx",
    ".cpp": "cxx",
    ".cxx": "cxx",
    ".s": "cc",
    ".S": "cc",
}


def Compilable(filename):
    """Return true if the file is compilable (should be in OBJS)."""
    for res in (filename.endswith(e) for e in COMPILABLE_EXTENSIONS):
        if res:
            return True
    return False


def Linkable(filename):
    """Return true if the file is linkable (should be on the link line)."""
    return filename.endswith(".o")


def Target(filename):
    """Translate a compilable filename to its .o target."""
    return os.path.splitext(filename)[0] + ".o"


def EscapeShellArgument(s):
    """Quotes an argument so that it will be interpreted literally by a POSIX
    shell. Taken from
    http://stackoverflow.com/questions/35817/whats-the-best-way-to-escape-ossystem-calls-in-python
    """
    return "'" + s.replace("'", "'\\''") + "'"


def EscapeMakeVariableExpansion(s):
    """Make has its own variable expansion syntax using $. We must escape it for
    string to be interpreted literally."""
    return s.replace("$", "$$")


def EscapeCppDefine(s):
    """Escapes a CPP define so that it will reach the compiler unaltered."""
    s = EscapeShellArgument(s)
    s = EscapeMakeVariableExpansion(s)
    # '#' characters must be escaped even embedded in a string, else Make will
    # treat it as the start of a comment.
    return s.replace("#", r"\#")


def QuoteIfNecessary(string):
    """TODO: Should this ideally be replaced with one or more of the above
    functions?"""
    if '"' in string:
        string = '"' + string.replace('"', '\\"') + '"'
    return string


def StringToMakefileVariable(string):
    """Convert a string to a value that is acceptable as a make variable name."""
    return re.sub("[^a-zA-Z0-9_]", "_", string)


srcdir_prefix = ""


def Sourceify(path):
    """Convert a path to its source directory form."""
    if "$(" in path:
        return path
    if os.path.isabs(path):
        return path
    return srcdir_prefix + path


def QuoteSpaces(s, quote=r"\ "):
    return s.replace(" ", quote)


def SourceifyAndQuoteSpaces(path):
    """Convert a path to its source directory form and quote spaces."""
    return QuoteSpaces(Sourceify(path))


# Map from qualified target to path to output.
target_outputs = {}
# Map from qualified target to any linkable output.  A subset
# of target_outputs.  E.g. when mybinary depends on liba, we want to
# include liba in the linker line; when otherbinary depends on
# mybinary, we just want to build mybinary first.
target_link_deps = {}


class MakefileWriter:
    """MakefileWriter packages up the writing of one target-specific foobar.mk.

    Its only real entry point is Write(), and is mostly used for namespacing.
    """

    def __init__(self, generator_flags, flavor):
        self.generator_flags = generator_flags
        self.flavor = flavor

        self.suffix_rules_srcdir = {}
        self.suffix_rules_objdir1 = {}
        self.suffix_rules_objdir2 = {}

        # Generate suffix rules for all compilable extensions.
        for ext in COMPILABLE_EXTENSIONS.keys():
            # Suffix rules for source folder.
            self.suffix_rules_srcdir.update(
                {
                    ext: (
                        """\
$(obj).$(TOOLSET)/$(TARGET)/%%.o: $(srcdir)/%%%s FORCE_DO_CMD
\t@$(call do_cmd,%s,1)
"""
                        % (ext, COMPILABLE_EXTENSIONS[ext])
                    )
                }
            )

            # Suffix rules for generated source files.
            self.suffix_rules_objdir1.update(
                {
                    ext: (
                        """\
$(obj).$(TOOLSET)/$(TARGET)/%%.o: $(obj).$(TOOLSET)/%%%s FORCE_DO_CMD
\t@$(call do_cmd,%s,1)
"""
                        % (ext, COMPILABLE_EXTENSIONS[ext])
                    )
                }
            )
            self.suffix_rules_objdir2.update(
                {
                    ext: (
                        """\
$(obj).$(TOOLSET)/$(TARGET)/%%.o: $(obj)/%%%s FORCE_DO_CMD
\t@$(call do_cmd,%s,1)
"""
                        % (ext, COMPILABLE_EXTENSIONS[ext])
                    )
                }
            )

    def Write(
        self, qualified_target, base_path, output_filename, spec, configs, part_of_all
    ):
        """The main entry point: writes a .mk file for a single target.

        Arguments:
          qualified_target: target we're generating
          base_path: path relative to source root we're building in, used to resolve
                     target-relative paths
          output_filename: output .mk file name to write
          spec, configs: gyp info
          part_of_all: flag indicating this target is part of 'all'
        """
        gyp.common.EnsureDirExists(output_filename)

        self.fp = open(output_filename, "w")

        self.fp.write(header)

        self.qualified_target = qualified_target
        self.path = base_path
        self.target = spec["target_name"]
        self.type = spec["type"]
        self.toolset = spec["toolset"]

        self.is_mac_bundle = gyp.xcode_emulation.IsMacBundle(self.flavor, spec)
        if self.flavor == "mac":
            self.xcode_settings = gyp.xcode_emulation.XcodeSettings(spec)
        else:
            self.xcode_settings = None

        deps, link_deps = self.ComputeDeps(spec)

        # Some of the generation below can add extra output, sources, or
        # link dependencies.  All of the out params of the functions that
        # follow use names like extra_foo.
        extra_outputs = []
        extra_sources = []
        extra_link_deps = []
        extra_mac_bundle_resources = []
        mac_bundle_deps = []

        if self.is_mac_bundle:
            self.output = self.ComputeMacBundleOutput(spec)
            self.output_binary = self.ComputeMacBundleBinaryOutput(spec)
        else:
            self.output = self.output_binary = self.ComputeOutput(spec)

        self.is_standalone_static_library = bool(
            spec.get("standalone_static_library", 0)
        )
        self._INSTALLABLE_TARGETS = ("executable", "loadable_module", "shared_library")
        if self.is_standalone_static_library or self.type in self._INSTALLABLE_TARGETS:
            self.alias = os.path.basename(self.output)
            install_path = self._InstallableTargetInstallPath()
        else:
            self.alias = self.output
            install_path = self.output

        self.WriteLn("TOOLSET := " + self.toolset)
        self.WriteLn("TARGET := " + self.target)

        # Actions must come first, since they can generate more OBJs for use below.
        if "actions" in spec:
            self.WriteActions(
                spec["actions"],
                extra_sources,
                extra_outputs,
                extra_mac_bundle_resources,
                part_of_all,
            )

        # Rules must be early like actions.
        if "rules" in spec:
            self.WriteRules(
                spec["rules"],
                extra_sources,
                extra_outputs,
                extra_mac_bundle_resources,
                part_of_all,
            )

        if "copies" in spec:
            self.WriteCopies(spec["copies"], extra_outputs, part_of_all)

        # Bundle resources.
        if self.is_mac_bundle:
            all_mac_bundle_resources = (
                spec.get("mac_bundle_resources", []) + extra_mac_bundle_resources
            )
            self.WriteMacBundleResources(all_mac_bundle_resources, mac_bundle_deps)
            self.WriteMacInfoPlist(mac_bundle_deps)

        # Sources.
        all_sources = spec.get("sources", []) + extra_sources
        if all_sources:
            self.WriteSources(
                configs,
                deps,
                all_sources,
                extra_outputs,
                extra_link_deps,
                part_of_all,
                gyp.xcode_emulation.MacPrefixHeader(
                    self.xcode_settings,
                    lambda p: Sourceify(self.Absolutify(p)),
                    self.Pchify,
                ),
            )
            sources = [x for x in all_sources if Compilable(x)]
            if sources:
                self.WriteLn(SHARED_HEADER_SUFFIX_RULES_COMMENT1)
                extensions = {os.path.splitext(s)[1] for s in sources}
                for ext in extensions:
                    if ext in self.suffix_rules_srcdir:
                        self.WriteLn(self.suffix_rules_srcdir[ext])
                self.WriteLn(SHARED_HEADER_SUFFIX_RULES_COMMENT2)
                for ext in extensions:
                    if ext in self.suffix_rules_objdir1:
                        self.WriteLn(self.suffix_rules_objdir1[ext])
                for ext in extensions:
                    if ext in self.suffix_rules_objdir2:
                        self.WriteLn(self.suffix_rules_objdir2[ext])
                self.WriteLn("# End of this set of suffix rules")

                # Add dependency from bundle to bundle binary.
                if self.is_mac_bundle:
                    mac_bundle_deps.append(self.output_binary)

        self.WriteTarget(
            spec,
            configs,
            deps,
            extra_link_deps + link_deps,
            mac_bundle_deps,
            extra_outputs,
            part_of_all,
        )

        # Update global list of target outputs, used in dependency tracking.
        target_outputs[qualified_target] = install_path

        # Update global list of link dependencies.
        if self.type in ("static_library", "shared_library"):
            target_link_deps[qualified_target] = self.output_binary

        # Currently any versions have the same effect, but in future the behavior
        # could be different.
        if self.generator_flags.get("android_ndk_version", None):
            self.WriteAndroidNdkModuleRule(self.target, all_sources, link_deps)

        self.fp.close()

    def WriteSubMake(self, output_filename, makefile_path, targets, build_dir):
        """Write a "sub-project" Makefile.

        This is a small, wrapper Makefile that calls the top-level Makefile to build
        the targets from a single gyp file (i.e. a sub-project).

        Arguments:
          output_filename: sub-project Makefile name to write
          makefile_path: path to the top-level Makefile
          targets: list of "all" targets for this sub-project
          build_dir: build output directory, relative to the sub-project
        """
        gyp.common.EnsureDirExists(output_filename)
        self.fp = open(output_filename, "w")
        self.fp.write(header)
        # For consistency with other builders, put sub-project build output in the
        # sub-project dir (see test/subdirectory/gyptest-subdir-all.py).
        self.WriteLn(
            "export builddir_name ?= %s"
            % os.path.join(os.path.dirname(output_filename), build_dir)
        )
        self.WriteLn(".PHONY: all")
        self.WriteLn("all:")
        if makefile_path:
            makefile_path = " -C " + makefile_path
        self.WriteLn("\t$(MAKE){} {}".format(makefile_path, " ".join(targets)))
        self.fp.close()

    def WriteActions(
        self,
        actions,
        extra_sources,
        extra_outputs,
        extra_mac_bundle_resources,
        part_of_all,
    ):
        """Write Makefile code for any 'actions' from the gyp input.

        extra_sources: a list that will be filled in with newly generated source
                       files, if any
        extra_outputs: a list that will be filled in with any outputs of these
                       actions (used to make other pieces dependent on these
                       actions)
        part_of_all: flag indicating this target is part of 'all'
        """
        env = self.GetSortedXcodeEnv()
        for action in actions:
            name = StringToMakefileVariable(
                "{}_{}".format(self.qualified_target, action["action_name"])
            )
            self.WriteLn('### Rules for action "%s":' % action["action_name"])
            inputs = action["inputs"]
            outputs = action["outputs"]

            # Build up a list of outputs.
            # Collect the output dirs we'll need.
            dirs = set()
            for out in outputs:
                dir = os.path.split(out)[0]
                if dir:
                    dirs.add(dir)
            if int(action.get("process_outputs_as_sources", False)):
                extra_sources += outputs
            if int(action.get("process_outputs_as_mac_bundle_resources", False)):
                extra_mac_bundle_resources += outputs

            # Write the actual command.
            action_commands = action["action"]
            if self.flavor == "mac":
                action_commands = [
                    gyp.xcode_emulation.ExpandEnvVars(command, env)
                    for command in action_commands
                ]
            command = gyp.common.EncodePOSIXShellList(action_commands)
            if "message" in action:
                self.WriteLn(
                    "quiet_cmd_{} = ACTION {} $@".format(name, action["message"])
                )
            else:
                self.WriteLn(f"quiet_cmd_{name} = ACTION {name} $@")
            if len(dirs) > 0:
                command = "mkdir -p %s" % " ".join(dirs) + "; " + command

            cd_action = "cd %s; " % Sourceify(self.path or ".")

            # command and cd_action get written to a toplevel variable called
            # cmd_foo. Toplevel variables can't handle things that change per
            # makefile like $(TARGET), so hardcode the target.
            command = command.replace("$(TARGET)", self.target)
            cd_action = cd_action.replace("$(TARGET)", self.target)

            # Set LD_LIBRARY_PATH in case the action runs an executable from this
            # build which links to shared libs from this build.
            # actions run on the host, so they should in theory only use host
            # libraries, but until everything is made cross-compile safe, also use
            # target libraries.
            # TODO(piman): when everything is cross-compile safe, remove lib.target
            self.WriteLn(
                "cmd_%s = LD_LIBRARY_PATH=$(builddir)/lib.host:"
                "$(builddir)/lib.target:$$LD_LIBRARY_PATH; "
                "export LD_LIBRARY_PATH; "
                "%s%s" % (name, cd_action, command)
            )
            self.WriteLn()
            outputs = [self.Absolutify(o) for o in outputs]
            # The makefile rules are all relative to the top dir, but the gyp actions
            # are defined relative to their containing dir.  This replaces the obj
            # variable for the action rule with an absolute version so that the output
            # goes in the right place.
            # Only write the 'obj' and 'builddir' rules for the "primary" output (:1);
            # it's superfluous for the "extra outputs", and this avoids accidentally
            # writing duplicate dummy rules for those outputs.
            # Same for environment.
            self.WriteLn("%s: obj := $(abs_obj)" % QuoteSpaces(outputs[0]))
            self.WriteLn("%s: builddir := $(abs_builddir)" % QuoteSpaces(outputs[0]))
            self.WriteSortedXcodeEnv(outputs[0], self.GetSortedXcodeEnv())

            for input in inputs:
                assert " " not in input, (
                    "Spaces in action input filenames not supported (%s)" % input
                )
            for output in outputs:
                assert " " not in output, (
                    "Spaces in action output filenames not supported (%s)" % output
                )

            # See the comment in WriteCopies about expanding env vars.
            outputs = [gyp.xcode_emulation.ExpandEnvVars(o, env) for o in outputs]
            inputs = [gyp.xcode_emulation.ExpandEnvVars(i, env) for i in inputs]

            self.WriteDoCmd(
                outputs,
                [Sourceify(self.Absolutify(i)) for i in inputs],
                part_of_all=part_of_all,
                command=name,
            )

            # Stuff the outputs in a variable so we can refer to them later.
            outputs_variable = "action_%s_outputs" % name
            self.WriteLn("{} := {}".format(outputs_variable, " ".join(outputs)))
            extra_outputs.append("$(%s)" % outputs_variable)
            self.WriteLn()

        self.WriteLn()

    def WriteRules(
        self,
        rules,
        extra_sources,
        extra_outputs,
        extra_mac_bundle_resources,
        part_of_all,
    ):
        """Write Makefile code for any 'rules' from the gyp input.

        extra_sources: a list that will be filled in with newly generated source
                       files, if any
        extra_outputs: a list that will be filled in with any outputs of these
                       rules (used to make other pieces dependent on these rules)
        part_of_all: flag indicating this target is part of 'all'
        """
        env = self.GetSortedXcodeEnv()
        for rule in rules:
            name = StringToMakefileVariable(
                "{}_{}".format(self.qualified_target, rule["rule_name"])
            )
            count = 0
            self.WriteLn("### Generated for rule %s:" % name)

            all_outputs = []

            for rule_source in rule.get("rule_sources", []):
                dirs = set()
                (rule_source_dirname, rule_source_basename) = os.path.split(rule_source)
                (rule_source_root, rule_source_ext) = os.path.splitext(
                    rule_source_basename
                )

                outputs = [
                    self.ExpandInputRoot(out, rule_source_root, rule_source_dirname)
                    for out in rule["outputs"]
                ]

                for out in outputs:
                    dir = os.path.dirname(out)
                    if dir:
                        dirs.add(dir)
                if int(rule.get("process_outputs_as_sources", False)):
                    extra_sources += outputs
                if int(rule.get("process_outputs_as_mac_bundle_resources", False)):
                    extra_mac_bundle_resources += outputs
                inputs = [
                    Sourceify(self.Absolutify(i))
                    for i in [rule_source] + rule.get("inputs", [])
                ]
                actions = ["$(call do_cmd,%s_%d)" % (name, count)]

                if name == "resources_grit":
                    # HACK: This is ugly.  Grit intentionally doesn't touch the
                    # timestamp of its output file when the file doesn't change,
                    # which is fine in hash-based dependency systems like scons
                    # and forge, but not kosher in the make world.  After some
                    # discussion, hacking around it here seems like the least
                    # amount of pain.
                    actions += ["@touch --no-create $@"]

                # See the comment in WriteCopies about expanding env vars.
                outputs = [gyp.xcode_emulation.ExpandEnvVars(o, env) for o in outputs]
                inputs = [gyp.xcode_emulation.ExpandEnvVars(i, env) for i in inputs]

                outputs = [self.Absolutify(o) for o in outputs]
                all_outputs += outputs
                # Only write the 'obj' and 'builddir' rules for the "primary" output
                # (:1); it's superfluous for the "extra outputs", and this avoids
                # accidentally writing duplicate dummy rules for those outputs.
                self.WriteLn("%s: obj := $(abs_obj)" % outputs[0])
                self.WriteLn("%s: builddir := $(abs_builddir)" % outputs[0])
                self.WriteMakeRule(
                    outputs, inputs, actions, command="%s_%d" % (name, count)
                )
                # Spaces in rule filenames are not supported, but rule variables have
                # spaces in them (e.g. RULE_INPUT_PATH expands to '$(abspath $<)').
                # The spaces within the variables are valid, so remove the variables
                # before checking.
                variables_with_spaces = re.compile(r"\$\([^ ]* \$<\)")
                for output in outputs:
                    output = re.sub(variables_with_spaces, "", output)
                    assert " " not in output, (
                        "Spaces in rule filenames not yet supported (%s)" % output
                    )
                self.WriteLn("all_deps += %s" % " ".join(outputs))

                action = [
                    self.ExpandInputRoot(ac, rule_source_root, rule_source_dirname)
                    for ac in rule["action"]
                ]
                mkdirs = ""
                if len(dirs) > 0:
                    mkdirs = "mkdir -p %s; " % " ".join(dirs)
                cd_action = "cd %s; " % Sourceify(self.path or ".")

                # action, cd_action, and mkdirs get written to a toplevel variable
                # called cmd_foo. Toplevel variables can't handle things that change
                # per makefile like $(TARGET), so hardcode the target.
                if self.flavor == "mac":
                    action = [
                        gyp.xcode_emulation.ExpandEnvVars(command, env)
                        for command in action
                    ]
                action = gyp.common.EncodePOSIXShellList(action)
                action = action.replace("$(TARGET)", self.target)
                cd_action = cd_action.replace("$(TARGET)", self.target)
                mkdirs = mkdirs.replace("$(TARGET)", self.target)

                # Set LD_LIBRARY_PATH in case the rule runs an executable from this
                # build which links to shared libs from this build.
                # rules run on the host, so they should in theory only use host
                # libraries, but until everything is made cross-compile safe, also use
                # target libraries.
                # TODO(piman): when everything is cross-compile safe, remove lib.target
                self.WriteLn(
                    "cmd_%(name)s_%(count)d = LD_LIBRARY_PATH="
                    "$(builddir)/lib.host:$(builddir)/lib.target:$$LD_LIBRARY_PATH; "
                    "export LD_LIBRARY_PATH; "
                    "%(cd_action)s%(mkdirs)s%(action)s"
                    % {
                        "action": action,
                        "cd_action": cd_action,
                        "count": count,
                        "mkdirs": mkdirs,
                        "name": name,
                    }
                )
                self.WriteLn(
                    "quiet_cmd_%(name)s_%(count)d = RULE %(name)s_%(count)d $@"
                    % {"count": count, "name": name}
                )
                self.WriteLn()
                count += 1

            outputs_variable = "rule_%s_outputs" % name
            self.WriteList(all_outputs, outputs_variable)
            extra_outputs.append("$(%s)" % outputs_variable)

            self.WriteLn("### Finished generating for rule: %s" % name)
            self.WriteLn()
        self.WriteLn("### Finished generating for all rules")
        self.WriteLn("")

    def WriteCopies(self, copies, extra_outputs, part_of_all):
        """Write Makefile code for any 'copies' from the gyp input.

        extra_outputs: a list that will be filled in with any outputs of this action
                       (used to make other pieces dependent on this action)
        part_of_all: flag indicating this target is part of 'all'
        """
        self.WriteLn("### Generated for copy rule.")

        variable = StringToMakefileVariable(self.qualified_target + "_copies")
        outputs = []
        for copy in copies:
            for path in copy["files"]:
                # Absolutify() may call normpath, and will strip trailing slashes.
                path = Sourceify(self.Absolutify(path))
                filename = os.path.split(path)[1]
                output = Sourceify(
                    self.Absolutify(os.path.join(copy["destination"], filename))
                )

                # If the output path has variables in it, which happens in practice for
                # 'copies', writing the environment as target-local doesn't work,
                # because the variables are already needed for the target name.
                # Copying the environment variables into global make variables doesn't
                # work either, because then the .d files will potentially contain spaces
                # after variable expansion, and .d file handling cannot handle spaces.
                # As a workaround, manually expand variables at gyp time. Since 'copies'
                # can't run scripts, there's no need to write the env then.
                # WriteDoCmd() will escape spaces for .d files.
                env = self.GetSortedXcodeEnv()
                output = gyp.xcode_emulation.ExpandEnvVars(output, env)
                path = gyp.xcode_emulation.ExpandEnvVars(path, env)
                self.WriteDoCmd([output], [path], "copy", part_of_all)
                outputs.append(output)
        self.WriteLn(
            "{} = {}".format(variable, " ".join(QuoteSpaces(o) for o in outputs))
        )
        extra_outputs.append("$(%s)" % variable)
        self.WriteLn()

    def WriteMacBundleResources(self, resources, bundle_deps):
        """Writes Makefile code for 'mac_bundle_resources'."""
        self.WriteLn("### Generated for mac_bundle_resources")

        for output, res in gyp.xcode_emulation.GetMacBundleResources(
            generator_default_variables["PRODUCT_DIR"],
            self.xcode_settings,
            [Sourceify(self.Absolutify(r)) for r in resources],
        ):
            _, ext = os.path.splitext(output)
            if ext != ".xcassets":
                # Make does not supports '.xcassets' emulation.
                self.WriteDoCmd(
                    [output], [res], "mac_tool,,,copy-bundle-resource", part_of_all=True
                )
                bundle_deps.append(output)

    def WriteMacInfoPlist(self, bundle_deps):
        """Write Makefile code for bundle Info.plist files."""
        info_plist, out, defines, extra_env = gyp.xcode_emulation.GetMacInfoPlist(
            generator_default_variables["PRODUCT_DIR"],
            self.xcode_settings,
            lambda p: Sourceify(self.Absolutify(p)),
        )
        if not info_plist:
            return
        if defines:
            # Create an intermediate file to store preprocessed results.
            intermediate_plist = "$(obj).$(TOOLSET)/$(TARGET)/" + os.path.basename(
                info_plist
            )
            self.WriteList(
                defines,
                intermediate_plist + ": INFOPLIST_DEFINES",
                "-D",
                quoter=EscapeCppDefine,
            )
            self.WriteMakeRule(
                [intermediate_plist],
                [info_plist],
                [
                    "$(call do_cmd,infoplist)",
                    # "Convert" the plist so that any weird whitespace changes from the
                    # preprocessor do not affect the XML parser in mac_tool.
                    "@plutil -convert xml1 $@ $@",
                ],
            )
            info_plist = intermediate_plist
        # plists can contain envvars and substitute them into the file.
        self.WriteSortedXcodeEnv(
            out, self.GetSortedXcodeEnv(additional_settings=extra_env)
        )
        self.WriteDoCmd(
            [out], [info_plist], "mac_tool,,,copy-info-plist", part_of_all=True
        )
        bundle_deps.append(out)

    def WriteSources(
        self,
        configs,
        deps,
        sources,
        extra_outputs,
        extra_link_deps,
        part_of_all,
        precompiled_header,
    ):
        """Write Makefile code for any 'sources' from the gyp input.
        These are source files necessary to build the current target.

        configs, deps, sources: input from gyp.
        extra_outputs: a list of extra outputs this action should be dependent on;
                       used to serialize action/rules before compilation
        extra_link_deps: a list that will be filled in with any outputs of
                         compilation (to be used in link lines)
        part_of_all: flag indicating this target is part of 'all'
        """

        # Write configuration-specific variables for CFLAGS, etc.
        for configname in sorted(configs.keys()):
            config = configs[configname]
            self.WriteList(
                config.get("defines"),
                "DEFS_%s" % configname,
                prefix="-D",
                quoter=EscapeCppDefine,
            )

            if self.flavor == "mac":
                cflags = self.xcode_settings.GetCflags(
                    configname, arch=config.get("xcode_configuration_platform")
                )
                cflags_c = self.xcode_settings.GetCflagsC(configname)
                cflags_cc = self.xcode_settings.GetCflagsCC(configname)
                cflags_objc = self.xcode_settings.GetCflagsObjC(configname)
                cflags_objcc = self.xcode_settings.GetCflagsObjCC(configname)
            else:
                cflags = config.get("cflags")
                cflags_c = config.get("cflags_c")
                cflags_cc = config.get("cflags_cc")

            self.WriteLn("# Flags passed to all source files.")
            self.WriteList(cflags, "CFLAGS_%s" % configname)
            self.WriteLn("# Flags passed to only C files.")
            self.WriteList(cflags_c, "CFLAGS_C_%s" % configname)
            self.WriteLn("# Flags passed to only C++ files.")
            self.WriteList(cflags_cc, "CFLAGS_CC_%s" % configname)
            if self.flavor == "mac":
                self.WriteLn("# Flags passed to only ObjC files.")
                self.WriteList(cflags_objc, "CFLAGS_OBJC_%s" % configname)
                self.WriteLn("# Flags passed to only ObjC++ files.")
                self.WriteList(cflags_objcc, "CFLAGS_OBJCC_%s" % configname)
            includes = config.get("include_dirs")
            if includes:
                includes = [Sourceify(self.Absolutify(i)) for i in includes]
            self.WriteList(includes, "INCS_%s" % configname, prefix="-I")

        compilable = list(filter(Compilable, sources))
        objs = [self.Objectify(self.Absolutify(Target(c))) for c in compilable]
        self.WriteList(objs, "OBJS")

        for obj in objs:
            assert " " not in obj, "Spaces in object filenames not supported (%s)" % obj
        self.WriteLn(
            "# Add to the list of files we specially track " "dependencies for."
        )
        self.WriteLn("all_deps += $(OBJS)")
        self.WriteLn()

        # Make sure our dependencies are built first.
        if deps:
            self.WriteMakeRule(
                ["$(OBJS)"],
                deps,
                comment="Make sure our dependencies are built " "before any of us.",
                order_only=True,
            )

        # Make sure the actions and rules run first.
        # If they generate any extra headers etc., the per-.o file dep tracking
        # will catch the proper rebuilds, so order only is still ok here.
        if extra_outputs:
            self.WriteMakeRule(
                ["$(OBJS)"],
                extra_outputs,
                comment="Make sure our actions/rules run " "before any of us.",
                order_only=True,
            )

        pchdeps = precompiled_header.GetObjDependencies(compilable, objs)
        if pchdeps:
            self.WriteLn("# Dependencies from obj files to their precompiled headers")
            for source, obj, gch in pchdeps:
                self.WriteLn(f"{obj}: {gch}")
            self.WriteLn("# End precompiled header dependencies")

        if objs:
            extra_link_deps.append("$(OBJS)")
            self.WriteLn(
                """\
# CFLAGS et al overrides must be target-local.
# See "Target-specific Variable Values" in the GNU Make manual."""
            )
            self.WriteLn("$(OBJS): TOOLSET := $(TOOLSET)")
            self.WriteLn(
                "$(OBJS): GYP_CFLAGS := "
                "$(DEFS_$(BUILDTYPE)) "
                "$(INCS_$(BUILDTYPE)) "
                "%s " % precompiled_header.GetInclude("c") + "$(CFLAGS_$(BUILDTYPE)) "
                "$(CFLAGS_C_$(BUILDTYPE))"
            )
            self.WriteLn(
                "$(OBJS): GYP_CXXFLAGS := "
                "$(DEFS_$(BUILDTYPE)) "
                "$(INCS_$(BUILDTYPE)) "
                "%s " % precompiled_header.GetInclude("cc") + "$(CFLAGS_$(BUILDTYPE)) "
                "$(CFLAGS_CC_$(BUILDTYPE))"
            )
            if self.flavor == "mac":
                self.WriteLn(
                    "$(OBJS): GYP_OBJCFLAGS := "
                    "$(DEFS_$(BUILDTYPE)) "
                    "$(INCS_$(BUILDTYPE)) "
                    "%s " % precompiled_header.GetInclude("m")
                    + "$(CFLAGS_$(BUILDTYPE)) "
                    "$(CFLAGS_C_$(BUILDTYPE)) "
                    "$(CFLAGS_OBJC_$(BUILDTYPE))"
                )
                self.WriteLn(
                    "$(OBJS): GYP_OBJCXXFLAGS := "
                    "$(DEFS_$(BUILDTYPE)) "
                    "$(INCS_$(BUILDTYPE)) "
                    "%s " % precompiled_header.GetInclude("mm")
                    + "$(CFLAGS_$(BUILDTYPE)) "
                    "$(CFLAGS_CC_$(BUILDTYPE)) "
                    "$(CFLAGS_OBJCC_$(BUILDTYPE))"
                )

        self.WritePchTargets(precompiled_header.GetPchBuildCommands())

        # If there are any object files in our input file list, link them into our
        # output.
        extra_link_deps += [source for source in sources if Linkable(source)]

        self.WriteLn()

    def WritePchTargets(self, pch_commands):
        """Writes make rules to compile prefix headers."""
        if not pch_commands:
            return

        for gch, lang_flag, lang, input in pch_commands:
            extra_flags = {
                "c": "$(CFLAGS_C_$(BUILDTYPE))",
                "cc": "$(CFLAGS_CC_$(BUILDTYPE))",
                "m": "$(CFLAGS_C_$(BUILDTYPE)) $(CFLAGS_OBJC_$(BUILDTYPE))",
                "mm": "$(CFLAGS_CC_$(BUILDTYPE)) $(CFLAGS_OBJCC_$(BUILDTYPE))",
            }[lang]
            var_name = {
                "c": "GYP_PCH_CFLAGS",
                "cc": "GYP_PCH_CXXFLAGS",
                "m": "GYP_PCH_OBJCFLAGS",
                "mm": "GYP_PCH_OBJCXXFLAGS",
            }[lang]
            self.WriteLn(
                f"{gch}: {var_name} := {lang_flag} " + "$(DEFS_$(BUILDTYPE)) "
                "$(INCS_$(BUILDTYPE)) "
                "$(CFLAGS_$(BUILDTYPE)) " + extra_flags
            )

            self.WriteLn(f"{gch}: {input} FORCE_DO_CMD")
            self.WriteLn("\t@$(call do_cmd,pch_%s,1)" % lang)
            self.WriteLn("")
            assert " " not in gch, "Spaces in gch filenames not supported (%s)" % gch
            self.WriteLn("all_deps += %s" % gch)
            self.WriteLn("")

    def ComputeOutputBasename(self, spec):
        """Return the 'output basename' of a gyp spec.

        E.g., the loadable module 'foobar' in directory 'baz' will produce
          'libfoobar.so'
        """
        assert not self.is_mac_bundle

        if self.flavor == "mac" and self.type in (
            "static_library",
            "executable",
            "shared_library",
            "loadable_module",
        ):
            return self.xcode_settings.GetExecutablePath()

        target = spec["target_name"]
        target_prefix = ""
        target_ext = ""
        if self.type == "static_library":
            if target[:3] == "lib":
                target = target[3:]
            target_prefix = "lib"
            target_ext = ".a"
        elif self.type in ("loadable_module", "shared_library"):
            if target[:3] == "lib":
                target = target[3:]
            target_prefix = "lib"
            if self.flavor == "aix":
                target_ext = ".a"
            else:
                target_ext = ".so"
        elif self.type == "none":
            target = "%s.stamp" % target
        elif self.type != "executable":
            print(
                "ERROR: What output file should be generated?",
                "type",
                self.type,
                "target",
                target,
            )

        target_prefix = spec.get("product_prefix", target_prefix)
        target = spec.get("product_name", target)
        product_ext = spec.get("product_extension")
        if product_ext:
            target_ext = "." + product_ext

        return target_prefix + target + target_ext

    def _InstallImmediately(self):
        return (
            self.toolset == "target"
            and self.flavor == "mac"
            and self.type
            in ("static_library", "executable", "shared_library", "loadable_module")
        )

    def ComputeOutput(self, spec):
        """Return the 'output' (full output path) of a gyp spec.

        E.g., the loadable module 'foobar' in directory 'baz' will produce
          '$(obj)/baz/libfoobar.so'
        """
        assert not self.is_mac_bundle

        path = os.path.join("$(obj)." + self.toolset, self.path)
        if self.type == "executable" or self._InstallImmediately():
            path = "$(builddir)"
        path = spec.get("product_dir", path)
        return os.path.join(path, self.ComputeOutputBasename(spec))

    def ComputeMacBundleOutput(self, spec):
        """Return the 'output' (full output path) to a bundle output directory."""
        assert self.is_mac_bundle
        path = generator_default_variables["PRODUCT_DIR"]
        return os.path.join(path, self.xcode_settings.GetWrapperName())

    def ComputeMacBundleBinaryOutput(self, spec):
        """Return the 'output' (full output path) to the binary in a bundle."""
        path = generator_default_variables["PRODUCT_DIR"]
        return os.path.join(path, self.xcode_settings.GetExecutablePath())

    def ComputeDeps(self, spec):
        """Compute the dependencies of a gyp spec.

        Returns a tuple (deps, link_deps), where each is a list of
        filenames that will need to be put in front of make for either
        building (deps) or linking (link_deps).
        """
        deps = []
        link_deps = []
        if "dependencies" in spec:
            deps.extend(
                [
                    target_outputs[dep]
                    for dep in spec["dependencies"]
                    if target_outputs[dep]
                ]
            )
            for dep in spec["dependencies"]:
                if dep in target_link_deps:
                    link_deps.append(target_link_deps[dep])
            deps.extend(link_deps)
            # TODO: It seems we need to transitively link in libraries (e.g. -lfoo)?
            # This hack makes it work:
            # link_deps.extend(spec.get('libraries', []))
        return (gyp.common.uniquer(deps), gyp.common.uniquer(link_deps))

    def WriteDependencyOnExtraOutputs(self, target, extra_outputs):
        self.WriteMakeRule(
            [self.output_binary],
            extra_outputs,
            comment="Build our special outputs first.",
            order_only=True,
        )

    def WriteTarget(
        self, spec, configs, deps, link_deps, bundle_deps, extra_outputs, part_of_all
    ):
        """Write Makefile code to produce the final target of the gyp spec.

        spec, configs: input from gyp.
        deps, link_deps: dependency lists; see ComputeDeps()
        extra_outputs: any extra outputs that our target should depend on
        part_of_all: flag indicating this target is part of 'all'
        """

        self.WriteLn("### Rules for final target.")

        if extra_outputs:
            self.WriteDependencyOnExtraOutputs(self.output_binary, extra_outputs)
            self.WriteMakeRule(
                extra_outputs,
                deps,
                comment=("Preserve order dependency of " "special output on deps."),
                order_only=True,
            )

        target_postbuilds = {}
        if self.type != "none":
            for configname in sorted(configs.keys()):
                config = configs[configname]
                if self.flavor == "mac":
                    ldflags = self.xcode_settings.GetLdflags(
                        configname,
                        generator_default_variables["PRODUCT_DIR"],
                        lambda p: Sourceify(self.Absolutify(p)),
                        arch=config.get("xcode_configuration_platform"),
                    )

                    # TARGET_POSTBUILDS_$(BUILDTYPE) is added to postbuilds later on.
                    gyp_to_build = gyp.common.InvertRelativePath(self.path)
                    target_postbuild = self.xcode_settings.AddImplicitPostbuilds(
                        configname,
                        QuoteSpaces(
                            os.path.normpath(os.path.join(gyp_to_build, self.output))
                        ),
                        QuoteSpaces(
                            os.path.normpath(
                                os.path.join(gyp_to_build, self.output_binary)
                            )
                        ),
                    )
                    if target_postbuild:
                        target_postbuilds[configname] = target_postbuild
                else:
                    ldflags = config.get("ldflags", [])
                    # Compute an rpath for this output if needed.
                    if any(dep.endswith(".so") or ".so." in dep for dep in deps):
                        # We want to get the literal string "$ORIGIN"
                        # into the link command, so we need lots of escaping.
                        ldflags.append(r"-Wl,-rpath=\$$ORIGIN/")
                        ldflags.append(r"-Wl,-rpath-link=\$(builddir)/")
                library_dirs = config.get("library_dirs", [])
                ldflags += [("-L%s" % library_dir) for library_dir in library_dirs]
                self.WriteList(ldflags, "LDFLAGS_%s" % configname)
                if self.flavor == "mac":
                    self.WriteList(
                        self.xcode_settings.GetLibtoolflags(configname),
                        "LIBTOOLFLAGS_%s" % configname,
                    )
            libraries = spec.get("libraries")
            if libraries:
                # Remove duplicate entries
                libraries = gyp.common.uniquer(libraries)
                if self.flavor == "mac":
                    libraries = self.xcode_settings.AdjustLibraries(libraries)
            self.WriteList(libraries, "LIBS")
            self.WriteLn(
                "%s: GYP_LDFLAGS := $(LDFLAGS_$(BUILDTYPE))"
                % QuoteSpaces(self.output_binary)
            )
            self.WriteLn("%s: LIBS := $(LIBS)" % QuoteSpaces(self.output_binary))

            if self.flavor == "mac":
                self.WriteLn(
                    "%s: GYP_LIBTOOLFLAGS := $(LIBTOOLFLAGS_$(BUILDTYPE))"
                    % QuoteSpaces(self.output_binary)
                )

        # Postbuild actions. Like actions, but implicitly depend on the target's
        # output.
        postbuilds = []
        if self.flavor == "mac":
            if target_postbuilds:
                postbuilds.append("$(TARGET_POSTBUILDS_$(BUILDTYPE))")
            postbuilds.extend(gyp.xcode_emulation.GetSpecPostbuildCommands(spec))

        if postbuilds:
            # Envvars may be referenced by TARGET_POSTBUILDS_$(BUILDTYPE),
            # so we must output its definition first, since we declare variables
            # using ":=".
            self.WriteSortedXcodeEnv(self.output, self.GetSortedXcodePostbuildEnv())

            for configname in target_postbuilds:
                self.WriteLn(
                    "%s: TARGET_POSTBUILDS_%s := %s"
                    % (
                        QuoteSpaces(self.output),
                        configname,
                        gyp.common.EncodePOSIXShellList(target_postbuilds[configname]),
                    )
                )

            # Postbuilds expect to be run in the gyp file's directory, so insert an
            # implicit postbuild to cd to there.
            postbuilds.insert(0, gyp.common.EncodePOSIXShellList(["cd", self.path]))
            for i, postbuild in enumerate(postbuilds):
                if not postbuild.startswith("$"):
                    postbuilds[i] = EscapeShellArgument(postbuild)
            self.WriteLn("%s: builddir := $(abs_builddir)" % QuoteSpaces(self.output))
            self.WriteLn(
                "%s: POSTBUILDS := %s"
                % (QuoteSpaces(self.output), " ".join(postbuilds))
            )

        # A bundle directory depends on its dependencies such as bundle resources
        # and bundle binary. When all dependencies have been built, the bundle
        # needs to be packaged.
        if self.is_mac_bundle:
            # If the framework doesn't contain a binary, then nothing depends
            # on the actions -- make the framework depend on them directly too.
            self.WriteDependencyOnExtraOutputs(self.output, extra_outputs)

            # Bundle dependencies. Note that the code below adds actions to this
            # target, so if you move these two lines, move the lines below as well.
            self.WriteList([QuoteSpaces(dep) for dep in bundle_deps], "BUNDLE_DEPS")
            self.WriteLn("%s: $(BUNDLE_DEPS)" % QuoteSpaces(self.output))

            # After the framework is built, package it. Needs to happen before
            # postbuilds, since postbuilds depend on this.
            if self.type in ("shared_library", "loadable_module"):
                self.WriteLn(
                    "\t@$(call do_cmd,mac_package_framework,,,%s)"
                    % self.xcode_settings.GetFrameworkVersion()
                )

            # Bundle postbuilds can depend on the whole bundle, so run them after
            # the bundle is packaged, not already after the bundle binary is done.
            if postbuilds:
                self.WriteLn("\t@$(call do_postbuilds)")
            postbuilds = []  # Don't write postbuilds for target's output.

            # Needed by test/mac/gyptest-rebuild.py.
            self.WriteLn("\t@true  # No-op, used by tests")

            # Since this target depends on binary and resources which are in
            # nested subfolders, the framework directory will be older than
            # its dependencies usually. To prevent this rule from executing
            # on every build (expensive, especially with postbuilds), expliclity
            # update the time on the framework directory.
            self.WriteLn("\t@touch -c %s" % QuoteSpaces(self.output))

        if postbuilds:
            assert not self.is_mac_bundle, (
                "Postbuilds for bundles should be done "
                "on the bundle, not the binary (target '%s')" % self.target
            )
            assert "product_dir" not in spec, (
                "Postbuilds do not work with " "custom product_dir"
            )

        if self.type == "executable":
            self.WriteLn(
                "%s: LD_INPUTS := %s"
                % (
                    QuoteSpaces(self.output_binary),
                    " ".join(QuoteSpaces(dep) for dep in link_deps),
                )
            )
            if self.toolset == "host" and self.flavor == "android":
                self.WriteDoCmd(
                    [self.output_binary],
                    link_deps,
                    "link_host",
                    part_of_all,
                    postbuilds=postbuilds,
                )
            else:
                self.WriteDoCmd(
                    [self.output_binary],
                    link_deps,
                    "link",
                    part_of_all,
                    postbuilds=postbuilds,
                )

        elif self.type == "static_library":
            for link_dep in link_deps:
                assert " " not in link_dep, (
                    "Spaces in alink input filenames not supported (%s)" % link_dep
                )
            if (
                self.flavor not in ("mac", "openbsd", "netbsd", "win")
                and not self.is_standalone_static_library
            ):
                self.WriteDoCmd(
                    [self.output_binary],
                    link_deps,
                    "alink_thin",
                    part_of_all,
                    postbuilds=postbuilds,
                )
            else:
                self.WriteDoCmd(
                    [self.output_binary],
                    link_deps,
                    "alink",
                    part_of_all,
                    postbuilds=postbuilds,
                )
        elif self.type == "shared_library":
            self.WriteLn(
                "%s: LD_INPUTS := %s"
                % (
                    QuoteSpaces(self.output_binary),
                    " ".join(QuoteSpaces(dep) for dep in link_deps),
                )
            )
            self.WriteDoCmd(
                [self.output_binary],
                link_deps,
                "solink",
                part_of_all,
                postbuilds=postbuilds,
            )
        elif self.type == "loadable_module":
            for link_dep in link_deps:
                assert " " not in link_dep, (
                    "Spaces in module input filenames not supported (%s)" % link_dep
                )
            if self.toolset == "host" and self.flavor == "android":
                self.WriteDoCmd(
                    [self.output_binary],
                    link_deps,
                    "solink_module_host",
                    part_of_all,
                    postbuilds=postbuilds,
                )
            else:
                self.WriteDoCmd(
                    [self.output_binary],
                    link_deps,
                    "solink_module",
                    part_of_all,
                    postbuilds=postbuilds,
                )
        elif self.type == "none":
            # Write a stamp line.
            self.WriteDoCmd(
                [self.output_binary], deps, "touch", part_of_all, postbuilds=postbuilds
            )
        else:
            print("WARNING: no output for", self.type, self.target)

        # Add an alias for each target (if there are any outputs).
        # Installable target aliases are created below.
        if (self.output and self.output != self.target) and (
            self.type not in self._INSTALLABLE_TARGETS
        ):
            self.WriteMakeRule(
                [self.target], [self.output], comment="Add target alias", phony=True
            )
            if part_of_all:
                self.WriteMakeRule(
                    ["all"],
                    [self.target],
                    comment='Add target alias to "all" target.',
                    phony=True,
                )

        # Add special-case rules for our installable targets.
        # 1) They need to install to the build dir or "product" dir.
        # 2) They get shortcuts for building (e.g. "make chrome").
        # 3) They are part of "make all".
        if self.type in self._INSTALLABLE_TARGETS or self.is_standalone_static_library:
            if self.type == "shared_library":
                file_desc = "shared library"
            elif self.type == "static_library":
                file_desc = "static library"
            else:
                file_desc = "executable"
            install_path = self._InstallableTargetInstallPath()
            installable_deps = [self.output]
            if (
                self.flavor == "mac"
                and "product_dir" not in spec
                and self.toolset == "target"
            ):
                # On mac, products are created in install_path immediately.
                assert install_path == self.output, "{} != {}".format(
                    install_path,
                    self.output,
                )

            # Point the target alias to the final binary output.
            self.WriteMakeRule(
                [self.target], [install_path], comment="Add target alias", phony=True
            )
            if install_path != self.output:
                assert not self.is_mac_bundle  # See comment a few lines above.
                self.WriteDoCmd(
                    [install_path],
                    [self.output],
                    "copy",
                    comment="Copy this to the %s output path." % file_desc,
                    part_of_all=part_of_all,
                )
                installable_deps.append(install_path)
            if self.output != self.alias and self.alias != self.target:
                self.WriteMakeRule(
                    [self.alias],
                    installable_deps,
                    comment="Short alias for building this %s." % file_desc,
                    phony=True,
                )
            if part_of_all:
                self.WriteMakeRule(
                    ["all"],
                    [install_path],
                    comment='Add %s to "all" target.' % file_desc,
                    phony=True,
                )

    def WriteList(self, value_list, variable=None, prefix="", quoter=QuoteIfNecessary):
        """Write a variable definition that is a list of values.

        E.g. WriteList(['a','b'], 'foo', prefix='blah') writes out
             foo = blaha blahb
        but in a pretty-printed style.
        """
        values = ""
        if value_list:
            value_list = [quoter(prefix + value) for value in value_list]
            values = " \\\n\t" + " \\\n\t".join(value_list)
        self.fp.write(f"{variable} :={values}\n\n")

    def WriteDoCmd(
        self, outputs, inputs, command, part_of_all, comment=None, postbuilds=False
    ):
        """Write a Makefile rule that uses do_cmd.

        This makes the outputs dependent on the command line that was run,
        as well as support the V= make command line flag.
        """
        suffix = ""
        if postbuilds:
            assert "," not in command
            suffix = ",,1"  # Tell do_cmd to honor $POSTBUILDS
        self.WriteMakeRule(
            outputs,
            inputs,
            actions=[f"$(call do_cmd,{command}{suffix})"],
            comment=comment,
            command=command,
            force=True,
        )
        # Add our outputs to the list of targets we read depfiles from.
        # all_deps is only used for deps file reading, and for deps files we replace
        # spaces with ? because escaping doesn't work with make's $(sort) and
        # other functions.
        outputs = [QuoteSpaces(o, SPACE_REPLACEMENT) for o in outputs]
        self.WriteLn("all_deps += %s" % " ".join(outputs))

    def WriteMakeRule(
        self,
        outputs,
        inputs,
        actions=None,
        comment=None,
        order_only=False,
        force=False,
        phony=False,
        command=None,
    ):
        """Write a Makefile rule, with some extra tricks.

        outputs: a list of outputs for the rule (note: this is not directly
                 supported by make; see comments below)
        inputs: a list of inputs for the rule
        actions: a list of shell commands to run for the rule
        comment: a comment to put in the Makefile above the rule (also useful
                 for making this Python script's code self-documenting)
        order_only: if true, makes the dependency order-only
        force: if true, include FORCE_DO_CMD as an order-only dep
        phony: if true, the rule does not actually generate the named output, the
               output is just a name to run the rule
        command: (optional) command name to generate unambiguous labels
        """
        outputs = [QuoteSpaces(o) for o in outputs]
        inputs = [QuoteSpaces(i) for i in inputs]

        if comment:
            self.WriteLn("# " + comment)
        if phony:
            self.WriteLn(".PHONY: " + " ".join(outputs))
        if actions:
            self.WriteLn("%s: TOOLSET := $(TOOLSET)" % outputs[0])
        force_append = " FORCE_DO_CMD" if force else ""

        if order_only:
            # Order only rule: Just write a simple rule.
            # TODO(evanm): just make order_only a list of deps instead of this hack.
            self.WriteLn(
                "{}: | {}{}".format(" ".join(outputs), " ".join(inputs), force_append)
            )
        elif len(outputs) == 1:
            # Regular rule, one output: Just write a simple rule.
            self.WriteLn("{}: {}{}".format(outputs[0], " ".join(inputs), force_append))
        else:
            # Regular rule, more than one output: Multiple outputs are tricky in
            # make. We will write three rules:
            # - All outputs depend on an intermediate file.
            # - Make .INTERMEDIATE depend on the intermediate.
            # - The intermediate file depends on the inputs and executes the
            #   actual command.
            # - The intermediate recipe will 'touch' the intermediate file.
            # - The multi-output rule will have an do-nothing recipe.

            # Hash the target name to avoid generating overlong filenames.
            cmddigest = hashlib.sha1(
                (command or self.target).encode("utf-8")
            ).hexdigest()
            intermediate = "%s.intermediate" % cmddigest
            self.WriteLn("{}: {}".format(" ".join(outputs), intermediate))
            self.WriteLn("\t%s" % "@:")
            self.WriteLn("{}: {}".format(".INTERMEDIATE", intermediate))
            self.WriteLn(
                "{}: {}{}".format(intermediate, " ".join(inputs), force_append)
            )
            actions.insert(0, "$(call do_cmd,touch)")

        if actions:
            for action in actions:
                self.WriteLn("\t%s" % action)
        self.WriteLn()

    def WriteAndroidNdkModuleRule(self, module_name, all_sources, link_deps):
        """Write a set of LOCAL_XXX definitions for Android NDK.

        These variable definitions will be used by Android NDK but do nothing for
        non-Android applications.

        Arguments:
          module_name: Android NDK module name, which must be unique among all
              module names.
          all_sources: A list of source files (will be filtered by Compilable).
          link_deps: A list of link dependencies, which must be sorted in
              the order from dependencies to dependents.
        """
        if self.type not in ("executable", "shared_library", "static_library"):
            return

        self.WriteLn("# Variable definitions for Android applications")
        self.WriteLn("include $(CLEAR_VARS)")
        self.WriteLn("LOCAL_MODULE := " + module_name)
        self.WriteLn(
            "LOCAL_CFLAGS := $(CFLAGS_$(BUILDTYPE)) "
            "$(DEFS_$(BUILDTYPE)) "
            # LOCAL_CFLAGS is applied to both of C and C++.  There is
            # no way to specify $(CFLAGS_C_$(BUILDTYPE)) only for C
            # sources.
            "$(CFLAGS_C_$(BUILDTYPE)) "
            # $(INCS_$(BUILDTYPE)) includes the prefix '-I' while
            # LOCAL_C_INCLUDES does not expect it.  So put it in
            # LOCAL_CFLAGS.
            "$(INCS_$(BUILDTYPE))"
        )
        # LOCAL_CXXFLAGS is obsolete and LOCAL_CPPFLAGS is preferred.
        self.WriteLn("LOCAL_CPPFLAGS := $(CFLAGS_CC_$(BUILDTYPE))")
        self.WriteLn("LOCAL_C_INCLUDES :=")
        self.WriteLn("LOCAL_LDLIBS := $(LDFLAGS_$(BUILDTYPE)) $(LIBS)")

        # Detect the C++ extension.
        cpp_ext = {".cc": 0, ".cpp": 0, ".cxx": 0}
        default_cpp_ext = ".cpp"
        for filename in all_sources:
            ext = os.path.splitext(filename)[1]
            if ext in cpp_ext:
                cpp_ext[ext] += 1
                if cpp_ext[ext] > cpp_ext[default_cpp_ext]:
                    default_cpp_ext = ext
        self.WriteLn("LOCAL_CPP_EXTENSION := " + default_cpp_ext)

        self.WriteList(
            list(map(self.Absolutify, filter(Compilable, all_sources))),
            "LOCAL_SRC_FILES",
        )

        # Filter out those which do not match prefix and suffix and produce
        # the resulting list without prefix and suffix.
        def DepsToModules(deps, prefix, suffix):
            modules = []
            for filepath in deps:
                filename = os.path.basename(filepath)
                if filename.startswith(prefix) and filename.endswith(suffix):
                    modules.append(filename[len(prefix) : -len(suffix)])
            return modules

        # Retrieve the default value of 'SHARED_LIB_SUFFIX'
        params = {"flavor": "linux"}
        default_variables = {}
        CalculateVariables(default_variables, params)

        self.WriteList(
            DepsToModules(
                link_deps,
                generator_default_variables["SHARED_LIB_PREFIX"],
                default_variables["SHARED_LIB_SUFFIX"],
            ),
            "LOCAL_SHARED_LIBRARIES",
        )
        self.WriteList(
            DepsToModules(
                link_deps,
                generator_default_variables["STATIC_LIB_PREFIX"],
                generator_default_variables["STATIC_LIB_SUFFIX"],
            ),
            "LOCAL_STATIC_LIBRARIES",
        )

        if self.type == "executable":
            self.WriteLn("include $(BUILD_EXECUTABLE)")
        elif self.type == "shared_library":
            self.WriteLn("include $(BUILD_SHARED_LIBRARY)")
        elif self.type == "static_library":
            self.WriteLn("include $(BUILD_STATIC_LIBRARY)")
        self.WriteLn()

    def WriteLn(self, text=""):
        self.fp.write(text + "\n")

    def GetSortedXcodeEnv(self, additional_settings=None):
        return gyp.xcode_emulation.GetSortedXcodeEnv(
            self.xcode_settings,
            "$(abs_builddir)",
            os.path.join("$(abs_srcdir)", self.path),
            "$(BUILDTYPE)",
            additional_settings,
        )

    def GetSortedXcodePostbuildEnv(self):
        # CHROMIUM_STRIP_SAVE_FILE is a chromium-specific hack.
        # TODO(thakis): It would be nice to have some general mechanism instead.
        strip_save_file = self.xcode_settings.GetPerTargetSetting(
            "CHROMIUM_STRIP_SAVE_FILE", ""
        )
        # Even if strip_save_file is empty, explicitly write it. Else a postbuild
        # might pick up an export from an earlier target.
        return self.GetSortedXcodeEnv(
            additional_settings={"CHROMIUM_STRIP_SAVE_FILE": strip_save_file}
        )

    def WriteSortedXcodeEnv(self, target, env):
        for k, v in env:
            # For
            #  foo := a\ b
            # the escaped space does the right thing. For
            #  export foo := a\ b
            # it does not -- the backslash is written to the env as literal character.
            # So don't escape spaces in |env[k]|.
            self.WriteLn(f"{QuoteSpaces(target)}: export {k} := {v}")

    def Objectify(self, path):
        """Convert a path to its output directory form."""
        if "$(" in path:
            path = path.replace("$(obj)/", "$(obj).%s/$(TARGET)/" % self.toolset)
        if "$(obj)" not in path:
            path = f"$(obj).{self.toolset}/$(TARGET)/{path}"
        return path

    def Pchify(self, path, lang):
        """Convert a prefix header path to its output directory form."""
        path = self.Absolutify(path)
        if "$(" in path:
            path = path.replace(
                "$(obj)/", f"$(obj).{self.toolset}/$(TARGET)/pch-{lang}"
            )
            return path
        return f"$(obj).{self.toolset}/$(TARGET)/pch-{lang}/{path}"

    def Absolutify(self, path):
        """Convert a subdirectory-relative path into a base-relative path.
        Skips over paths that contain variables."""
        if "$(" in path:
            # Don't call normpath in this case, as it might collapse the
            # path too aggressively if it features '..'. However it's still
            # important to strip trailing slashes.
            return path.rstrip("/")
        return os.path.normpath(os.path.join(self.path, path))

    def ExpandInputRoot(self, template, expansion, dirname):
        if "%(INPUT_ROOT)s" not in template and "%(INPUT_DIRNAME)s" not in template:
            return template
        path = template % {
            "INPUT_ROOT": expansion,
            "INPUT_DIRNAME": dirname,
        }
        return path

    def _InstallableTargetInstallPath(self):
        """Returns the location of the final output for an installable target."""
        # Functionality removed for all platforms to match Xcode and hoist
        # shared libraries into PRODUCT_DIR for users:
        # Xcode puts shared_library results into PRODUCT_DIR, and some gyp files
        # rely on this. Emulate this behavior for mac.
        # if self.type == "shared_library" and (
        #     self.flavor != "mac" or self.toolset != "target"
        # ):
        #    # Install all shared libs into a common directory (per toolset) for
        #    # convenient access with LD_LIBRARY_PATH.
        #    return "$(builddir)/lib.%s/%s" % (self.toolset, self.alias)
        return "$(builddir)/" + self.alias


def WriteAutoRegenerationRule(params, root_makefile, makefile_name, build_files):
    """Write the target to regenerate the Makefile."""
    options = params["options"]
    build_files_args = [
        gyp.common.RelativePath(filename, options.toplevel_dir)
        for filename in params["build_files_arg"]
    ]

    gyp_binary = gyp.common.FixIfRelativePath(
        params["gyp_binary"], options.toplevel_dir
    )
    if not gyp_binary.startswith(os.sep):
        gyp_binary = os.path.join(".", gyp_binary)

    root_makefile.write(
        "quiet_cmd_regen_makefile = ACTION Regenerating $@\n"
        "cmd_regen_makefile = cd $(srcdir); %(cmd)s\n"
        "%(makefile_name)s: %(deps)s\n"
        "\t$(call do_cmd,regen_makefile)\n\n"
        % {
            "makefile_name": makefile_name,
            "deps": " ".join(SourceifyAndQuoteSpaces(bf) for bf in build_files),
            "cmd": gyp.common.EncodePOSIXShellList(
                [gyp_binary, "-fmake"] + gyp.RegenerateFlags(options) + build_files_args
            ),
        }
    )


def PerformBuild(data, configurations, params):
    options = params["options"]
    for config in configurations:
        arguments = ["make"]
        if options.toplevel_dir and options.toplevel_dir != ".":
            arguments += "-C", options.toplevel_dir
        arguments.append("BUILDTYPE=" + config)
        print(f"Building [{config}]: {arguments}")
        subprocess.check_call(arguments)


def GenerateOutput(target_list, target_dicts, data, params):
    options = params["options"]
    flavor = gyp.common.GetFlavor(params)
    generator_flags = params.get("generator_flags", {})
    builddir_name = generator_flags.get("output_dir", "out")
    android_ndk_version = generator_flags.get("android_ndk_version", None)
    default_target = generator_flags.get("default_target", "all")

    def CalculateMakefilePath(build_file, base_name):
        """Determine where to write a Makefile for a given gyp file."""
        # Paths in gyp files are relative to the .gyp file, but we want
        # paths relative to the source root for the master makefile.  Grab
        # the path of the .gyp file as the base to relativize against.
        # E.g. "foo/bar" when we're constructing targets for "foo/bar/baz.gyp".
        base_path = gyp.common.RelativePath(os.path.dirname(build_file), options.depth)
        # We write the file in the base_path directory.
        output_file = os.path.join(options.depth, base_path, base_name)
        if options.generator_output:
            output_file = os.path.join(
                options.depth, options.generator_output, base_path, base_name
            )
        base_path = gyp.common.RelativePath(
            os.path.dirname(build_file), options.toplevel_dir
        )
        return base_path, output_file

    # TODO:  search for the first non-'Default' target.  This can go
    # away when we add verification that all targets have the
    # necessary configurations.
    default_configuration = None
    toolsets = {target_dicts[target]["toolset"] for target in target_list}
    for target in target_list:
        spec = target_dicts[target]
        if spec["default_configuration"] != "Default":
            default_configuration = spec["default_configuration"]
            break
    if not default_configuration:
        default_configuration = "Default"

    srcdir = "."
    makefile_name = "Makefile" + options.suffix
    makefile_path = os.path.join(options.toplevel_dir, makefile_name)
    if options.generator_output:
        global srcdir_prefix
        makefile_path = os.path.join(
            options.toplevel_dir, options.generator_output, makefile_name
        )
        srcdir = gyp.common.RelativePath(srcdir, options.generator_output)
        srcdir_prefix = "$(srcdir)/"

    flock_command = "flock"
    copy_archive_arguments = "-af"
    makedep_arguments = "-MMD"
    header_params = {
        "default_target": default_target,
        "builddir": builddir_name,
        "default_configuration": default_configuration,
        "flock": flock_command,
        "flock_index": 1,
        "link_commands": LINK_COMMANDS_LINUX,
        "extra_commands": "",
        "srcdir": srcdir,
        "copy_archive_args": copy_archive_arguments,
        "makedep_args": makedep_arguments,
        "CC.target": GetEnvironFallback(("CC_target", "CC"), "$(CC)"),
        "AR.target": GetEnvironFallback(("AR_target", "AR"), "$(AR)"),
        "CXX.target": GetEnvironFallback(("CXX_target", "CXX"), "$(CXX)"),
        "LINK.target": GetEnvironFallback(("LINK_target", "LINK"), "$(LINK)"),
        "CC.host": GetEnvironFallback(("CC_host", "CC"), "gcc"),
        "AR.host": GetEnvironFallback(("AR_host", "AR"), "ar"),
        "CXX.host": GetEnvironFallback(("CXX_host", "CXX"), "g++"),
        "LINK.host": GetEnvironFallback(("LINK_host", "LINK"), "$(CXX.host)"),
    }
    if flavor == "mac":
        flock_command = "./gyp-mac-tool flock"
        header_params.update(
            {
                "flock": flock_command,
                "flock_index": 2,
                "link_commands": LINK_COMMANDS_MAC,
                "extra_commands": SHARED_HEADER_MAC_COMMANDS,
            }
        )
    elif flavor == "android":
        header_params.update({"link_commands": LINK_COMMANDS_ANDROID})
    elif flavor == "zos":
        copy_archive_arguments = "-fPR"
        makedep_arguments = "-qmakedep=gcc"
        header_params.update(
            {
                "copy_archive_args": copy_archive_arguments,
                "makedep_args": makedep_arguments,
                "link_commands": LINK_COMMANDS_OS390,
                "CC.target": GetEnvironFallback(("CC_target", "CC"), "njsc"),
                "CXX.target": GetEnvironFallback(("CXX_target", "CXX"), "njsc++"),
                "CC.host": GetEnvironFallback(("CC_host", "CC"), "njsc"),
                "CXX.host": GetEnvironFallback(("CXX_host", "CXX"), "njsc++"),
            }
        )
    elif flavor == "solaris":
        copy_archive_arguments = "-pPRf@"
        header_params.update(
            {
                "copy_archive_args": copy_archive_arguments,
                "flock": "./gyp-flock-tool flock",
                "flock_index": 2,
            }
        )
    elif flavor == "freebsd":
        # Note: OpenBSD has sysutils/flock. lockf seems to be FreeBSD specific.
        header_params.update({"flock": "lockf"})
    elif flavor == "openbsd":
        copy_archive_arguments = "-pPRf"
        header_params.update({"copy_archive_args": copy_archive_arguments})
    elif flavor == "aix":
        copy_archive_arguments = "-pPRf"
        header_params.update(
            {
                "copy_archive_args": copy_archive_arguments,
                "link_commands": LINK_COMMANDS_AIX,
                "flock": "./gyp-flock-tool flock",
                "flock_index": 2,
            }
        )

    build_file, _, _ = gyp.common.ParseQualifiedTarget(target_list[0])
    make_global_settings_array = data[build_file].get("make_global_settings", [])
    wrappers = {}
    for key, value in make_global_settings_array:
        if key.endswith("_wrapper"):
            wrappers[key[: -len("_wrapper")]] = "$(abspath %s)" % value
    make_global_settings = ""
    for key, value in make_global_settings_array:
        if re.match(".*_wrapper", key):
            continue
        if value[0] != "$":
            value = "$(abspath %s)" % value
        wrapper = wrappers.get(key)
        if wrapper:
            value = f"{wrapper} {value}"
            del wrappers[key]
        if key in ("CC", "CC.host", "CXX", "CXX.host"):
            make_global_settings += (
                "ifneq (,$(filter $(origin %s), undefined default))\n" % key
            )
            # Let gyp-time envvars win over global settings.
            env_key = key.replace(".", "_")  # CC.host -> CC_host
            if env_key in os.environ:
                value = os.environ[env_key]
            make_global_settings += f"  {key} = {value}\n"
            make_global_settings += "endif\n"
        else:
            make_global_settings += f"{key} ?= {value}\n"
    # TODO(ukai): define cmd when only wrapper is specified in
    # make_global_settings.

    header_params["make_global_settings"] = make_global_settings

    gyp.common.EnsureDirExists(makefile_path)
    root_makefile = open(makefile_path, "w")
    root_makefile.write(SHARED_HEADER % header_params)
    # Currently any versions have the same effect, but in future the behavior
    # could be different.
    if android_ndk_version:
        root_makefile.write(
            "# Define LOCAL_PATH for build of Android applications.\n"
            "LOCAL_PATH := $(call my-dir)\n"
            "\n"
        )
    for toolset in toolsets:
        root_makefile.write("TOOLSET := %s\n" % toolset)
        WriteRootHeaderSuffixRules(root_makefile)

    # Put build-time support tools next to the root Makefile.
    dest_path = os.path.dirname(makefile_path)
    gyp.common.CopyTool(flavor, dest_path)

    # Find the list of targets that derive from the gyp file(s) being built.
    needed_targets = set()
    for build_file in params["build_files"]:
        for target in gyp.common.AllTargets(target_list, target_dicts, build_file):
            needed_targets.add(target)

    build_files = set()
    include_list = set()
    for qualified_target in target_list:
        build_file, target, toolset = gyp.common.ParseQualifiedTarget(qualified_target)

        this_make_global_settings = data[build_file].get("make_global_settings", [])
        assert make_global_settings_array == this_make_global_settings, (
            "make_global_settings needs to be the same for all targets "
            f"{this_make_global_settings} vs. {make_global_settings}"
        )

        build_files.add(gyp.common.RelativePath(build_file, options.toplevel_dir))
        included_files = data[build_file]["included_files"]
        for included_file in included_files:
            # The included_files entries are relative to the dir of the build file
            # that included them, so we have to undo that and then make them relative
            # to the root dir.
            relative_include_file = gyp.common.RelativePath(
                gyp.common.UnrelativePath(included_file, build_file),
                options.toplevel_dir,
            )
            abs_include_file = os.path.abspath(relative_include_file)
            # If the include file is from the ~/.gyp dir, we should use absolute path
            # so that relocating the src dir doesn't break the path.
            if params["home_dot_gyp"] and abs_include_file.startswith(
                params["home_dot_gyp"]
            ):
                build_files.add(abs_include_file)
            else:
                build_files.add(relative_include_file)

        base_path, output_file = CalculateMakefilePath(
            build_file, target + "." + toolset + options.suffix + ".mk"
        )

        spec = target_dicts[qualified_target]
        configs = spec["configurations"]

        if flavor == "mac":
            gyp.xcode_emulation.MergeGlobalXcodeSettingsToSpec(data[build_file], spec)

        writer = MakefileWriter(generator_flags, flavor)
        writer.Write(
            qualified_target,
            base_path,
            output_file,
            spec,
            configs,
            part_of_all=qualified_target in needed_targets,
        )

        # Our root_makefile lives at the source root.  Compute the relative path
        # from there to the output_file for including.
        mkfile_rel_path = gyp.common.RelativePath(
            output_file, os.path.dirname(makefile_path)
        )
        include_list.add(mkfile_rel_path)

    # Write out per-gyp (sub-project) Makefiles.
    depth_rel_path = gyp.common.RelativePath(options.depth, os.getcwd())
    for build_file in build_files:
        # The paths in build_files were relativized above, so undo that before
        # testing against the non-relativized items in target_list and before
        # calculating the Makefile path.
        build_file = os.path.join(depth_rel_path, build_file)
        gyp_targets = [
            target_dicts[qualified_target]["target_name"]
            for qualified_target in target_list
            if qualified_target.startswith(build_file)
            and qualified_target in needed_targets
        ]
        # Only generate Makefiles for gyp files with targets.
        if not gyp_targets:
            continue
        base_path, output_file = CalculateMakefilePath(
            build_file, os.path.splitext(os.path.basename(build_file))[0] + ".Makefile"
        )
        makefile_rel_path = gyp.common.RelativePath(
            os.path.dirname(makefile_path), os.path.dirname(output_file)
        )
        writer.WriteSubMake(output_file, makefile_rel_path, gyp_targets, builddir_name)

    # Write out the sorted list of includes.
    root_makefile.write("\n")
    for include_file in sorted(include_list):
        # We wrap each .mk include in an if statement so users can tell make to
        # not load a file by setting NO_LOAD.  The below make code says, only
        # load the .mk file if the .mk filename doesn't start with a token in
        # NO_LOAD.
        root_makefile.write(
            "ifeq ($(strip $(foreach prefix,$(NO_LOAD),\\\n"
            "    $(findstring $(join ^,$(prefix)),\\\n"
            "                 $(join ^," + include_file + ")))),)\n"
        )
        root_makefile.write("  include " + include_file + "\n")
        root_makefile.write("endif\n")
    root_makefile.write("\n")

    if not generator_flags.get("standalone") and generator_flags.get(
        "auto_regeneration", True
    ):
        WriteAutoRegenerationRule(params, root_makefile, makefile_name, build_files)

    root_makefile.write(SHARED_FOOTER)

    root_makefile.close()# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
from typing import Callable, Tuple

import torch
from torch import Tensor, device, dtype, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .file_utils import (
    DUMMY_INPUTS,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    cached_path,
    hf_bucket_url,
    is_remote_url,
)


logger = logging.getLogger(__name__)


try:
    from torch.nn import Identity
except ImportError:
    # Older PyTorch compatibility
    class Identity(nn.Module):
        r"""A placeholder identity operator that is argument-insensitive.
        """

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, input):
            return input


class ModuleUtilsMixin:
    """
    A few utilities for torch.nn.Modules, to be used as a mixin.
    """

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Get number of (optionally, trainable) parameters in the module.
        """
        params = filter(lambda x: x.requires_grad, self.parameters()) if only_trainable else self.parameters()
        return sum(p.numel() for p in params)

    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        try:
            import psutil
        except (ImportError):
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        try:
            import psutil
        except (ImportError):
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        return None

    def add_memory_hooks(self):
        """ Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.
            Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero with `model.reset_memory_hooks_state()`
        """
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        self.reset_memory_hooks_state()

    def reset_memory_hooks_state(self):
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

    @property
    def device(self) -> device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> dtype:
        return next(self.parameters()).dtype

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """type: torch.Tensor -> torch.Tensor"""
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        return encoder_extended_attention_mask

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: tuple, device: device):
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.

        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: torch.Device, usually self.device

        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        attention_probs has shape bsz x n_heads x N x N
        Arguments:
            head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
            num_hidden_layers: int
        Returns:
             Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
             or list with [None] for each layer
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to fload if need + fp16 compatibility
        return head_mask


class PreTrainedModel(nn.Module, ModuleUtilsMixin):
    r""" Base class for all models.

        :class:`~transformers.PreTrainedModel` takes care of storing the configuration of the models and handles methods for loading/downloading/saving models
        as well as a few methods common to all models to (i) resize the input embeddings and (ii) prune heads in the self-attention heads.

        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~transformers.PretrainedConfig` to use as configuration class for this model architecture.
            - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names` (string) as keys and `url` (string) of associated pretrained weights as values.
            - ``load_tf_weights``: a python ``method`` for loading a TensorFlow checkpoint in a PyTorch model, taking as arguments:

                - ``model``: an instance of the relevant subclass of :class:`~transformers.PreTrainedModel`,
                - ``config``: an instance of the relevant subclass of :class:`~transformers.PretrainedConfig`,
                - ``path``: a path (string) to the TensorFlow checkpoint.

            - ``base_model_prefix``: a string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
    """
    config_class = None
    pretrained_model_archive_map = {}
    base_model_prefix = ""

    @property
    def dummy_inputs(self):
        """ Dummy inputs to do a forward pass in the network.

        Returns:
            torch.Tensor with dummy inputs
        """
        return {"input_ids": torch.tensor(DUMMY_INPUTS)}

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    def get_input_embeddings(self):
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Module`:
                A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value):
        """
        Set model's input embeddings

        Args:
            value (:obj:`nn.Module`):
                A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        """
        Returns the model's output embeddings.

        Returns:
            :obj:`nn.Module`:
                A torch module mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone module weights depending of whether we are using TorchScript or not
        """
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
                If not provided or None: does nothing and just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.

        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        return self.get_input_embeddings()

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        # Tie weights if needed
        self.tie_weights()

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.

            Arguments:

                heads_to_prune: dict with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`).
                E.g. {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

        self.base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

            Arguments:
                save_directory: directory to which to save.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        if getattr(self.config, "xla_device", False):
            import torch_xla.core.xla_model as xm

            if xm.is_master_ordinal():
                # Save configuration file
                model_to_save.config.save_pretrained(save_directory)
            # xm.save takes care of saving only from master
            xm.save(model_to_save.state_dict(), output_model_file)
        else:
            model_to_save.config.save_pretrained(save_directory)
            torch.save(model_to_save.state_dict(), output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:
              - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
              - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
              - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
              - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
              - None if you are both providing the configuration and state dictionary (resp. with keyword arguments ``config`` and ``state_dict``)

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) one of:
                - an instance of a class derived from :class:`~transformers.PretrainedConfig`, or
                - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained()`
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                    - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                    - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                    - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            # For example purposes. Not runnable.
            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_cdn = kwargs.pop("use_cdn", True)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            elif os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    use_cdn=use_cdn,
                )

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
            except EnvironmentError:
                if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                    msg = "Couldn't reach server at '{}' to download pretrained weights.".format(archive_file)
                else:
                    msg = (
                        "Model name '{}' was not found in model name list ({}). "
                        "We assumed '{}' was a path or url to model weight files named one of {} but "
                        "couldn't find any such file at this path or url.".format(
                            pretrained_model_name_or_path,
                            ", ".join(cls.pretrained_model_archive_map.keys()),
                            archive_file,
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME],
                        )
                    )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                    "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module: nn.Module, prefix=""):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ""
            model_to_load = model
            has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
            if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
                start_prefix = cls.base_model_prefix + "."
            if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(model, cls.base_model_prefix)

            load(model_to_load, prefix=start_prefix)

            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
                ]

                missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

            if len(missing_keys) > 0:
                logger.info(
                    "Weights of {} not initialized from pretrained model: {}".format(
                        model.__class__.__name__, missing_keys
                    )
                )
            if len(unexpected_keys) > 0:
                logger.info(
                    "Weights from pretrained model not used in {}: {}".format(
                        model.__class__.__name__, unexpected_keys
                    )
                )
            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
        model.tie_weights()  # make sure token embedding weights are still tied if needed

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        if hasattr(config, "xla_device") and config.xla_device:
            import torch_xla.core.xla_model as xm

            model = xm.send_cpu_data_to_device(model, xm.xla_device())
            model = model.to(xm.xla_device())

        return model

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def prepare_scores_for_generation(self, scores, **kwargs):
        return scores

    def _use_cache(self, outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        if len(outputs) <= 1 or use_cache is False:
            return False
        if hasattr(self.config, "mem_len") and self.config.mem_len == 0:
            return False
        return True

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
        use_cache=None,
        **model_specific_kwargs
    ):
        r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.

            min_length: (`optional`) int
                The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            early_stopping: (`optional`) bool
                if set to `True` beam search is stopped when at least `num_beams` sentences finished per batch. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

            pad_token_id: (`optional`) int
                Padding token. Default to specicic model pad_token_id or None if it does not exist.

            bos_token_id: (`optional`) int
                BOS token. Defaults to `bos_token_id` as defined in the models config.

            eos_token_id: (`optional`) int
                EOS token. Defaults to `eos_token_id` as defined in the models config.

            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            no_repeat_ngram_size: (`optional`) int
                If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once.
            bad_words_ids: (`optional`) list of lists of int
                `bad_words_ids` contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.

            attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                Defaults to `None`.

            `What are attention masks? <../glossary.html#attention-mask>`__

            decoder_start_token_id=None: (`optional`) int
                If an encoder-decoder model starts decoding with a different token than BOS.
                Defaults to `None` and is changed to `BOS` later.

            use_cache: (`optional`) bool
                If `use_cache` is True, past key values are used to speed up decoding if applicable to model. Defaults to `True`.

            model_specific_kwargs: (`optional`) dict
                Additional model specific kwargs will be forwarded to the `forward` function of the model.

        Return:

            output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
                sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3)  # 3 generate sequences using by sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()
            if 'task_prompt_recored' in model_specific_kwargs:
                encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask,task_prompt_recored=model_specific_kwargs['task_prompt_recored'])
            else:
                encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)
        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]
        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        batch_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )
            outputs = self(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                self.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                next_token_logits[:, eos_token_id] = -float("inf")

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

            cur_len = cur_len + 1

        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids

        for hypo_idx, hypo in enumerate(input_ids):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        return decoded

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example with beam search.
        """

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        # done sentences
        done = [False for _ in range(batch_size)]
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                self.enforce_repetition_penalty_(
                    next_token_logits, batch_size, num_beams, input_ids, repetition_penalty,
                )

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
            if self.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solutino
                scores = self.prepare_scores_for_generation(scores, cur_len=cur_len, max_length=max_length)

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                scores[:, eos_token_id] = -float("inf")

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                num_batch_hypotheses = batch_size * num_beams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_batch_tokens = calc_banned_ngram_tokens(
                    input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
                )
                for i, banned_tokens in enumerate(banned_batch_tokens):
                    scores[i, banned_tokens] = -float("inf")

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                for i, banned_tokens in enumerate(banned_tokens):
                    scores[i, banned_tokens] = -float("inf")

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence or last iteration
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token if it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if were done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len=cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1)

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

            # update current length
            cur_len = cur_len + 1

        # finalize all open beam hypotheses and end to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() is not eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are filled with pad_token
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded

    @staticmethod
    def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)


def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class PoolerStartLogits(nn.Module):
    """ Compute SQuAD start_logits from sequence hidden states. """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerEndLogits(nn.Module):
    """ Compute SQuAD end_logits from sequence hidden states and start token hidden state.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None):
        """ Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to hidden_states
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
                Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerAnswerClass(nn.Module):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states, start_states=None, start_positions=None, cls_index=None):
        """
        Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to ``hidden_states``.
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span.
            **cls_index**: torch.LongTensor of shape ``(batch_size,)``
                position of the CLS token. If None, take the last token.

            note(Original repo):
                no dependency on end_feature so that we can obtain one single `cls_logits`
                for each sample
        """
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x


class SQuADHead(nn.Module):
    r""" A SQuAD head inspired by XLNet.

    Parameters:
        config (:class:`~transformers.XLNetConfig`): Model configuration class with all the parameters of the model.

    Inputs:
        **hidden_states**: ``torch.FloatTensor`` of shape ``(batch_size, seq_len, hidden_size)``
            hidden states of sequence tokens
        **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the first token for the labeled span.
        **end_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
            position of the last token for the labeled span.
        **cls_index**: torch.LongTensor of shape ``(batch_size,)``
            position of the CLS token. If None, take the last token.
        **is_impossible**: ``torch.LongTensor`` of shape ``(batch_size,)``
            Whether the question has a possible answer in the paragraph or not.
        **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
            Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
            1.0 means token should be masked.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned if both ``start_positions`` and ``end_positions`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.
        **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``
            Indices for the top config.start_n_top start token possibilities (beam-search).
        **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size,)``
            Log probabilities for the ``is_impossible`` label of the answers.
    """

    def __init__(self, config):
        super().__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def forward(
        self, hidden_states, start_positions=None, end_positions=None, cls_index=None, is_impossible=None, p_mask=None,
    ):
        outputs = ()

        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            outputs = (total_loss,) + outputs

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits,) + outputs

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        # or (if labels are provided) (total_loss,)
        return outputs


class SequenceSummary(nn.Module):
    r""" Compute a single vector summary of a sequence hidden states according to various possibilities:
        Args of the config class:
            summary_type:
                - 'last' => [default] take the last token hidden state (like XLNet)
                - 'first' => take the first token hidden state (like Bert)
                - 'mean' => take the mean of all tokens hidden states
                - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
                - 'attn' => Not implemented now, use multi-head attention
            summary_use_proj: Add a projection after the vector extraction
            summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
            summary_activation: 'tanh' or another string => add an activation to the output, Other => no activation. Default
            summary_first_dropout: Add a dropout before the projection and activation
            summary_last_dropout: Add a dropout after the projection and activation
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = (get_activation(activation_string) if activation_string else Identity())

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hidden_states, cls_index=None):
        """ hidden_states: float Tensor in shape [bsz, ..., seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(hidden_states[..., :1, :], hidden_states.shape[-2] - 1, dtype=torch.long,)
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def prune_conv1d_layer(layer, index, dim=1):
    """ Prune a Conv1D layer (a model parameters) to keep only entries in index.
        A Conv1D work as a Linear layer (see e.g. BERT) but the weights are transposed.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def prune_layer(layer, index, dim=None):
    """ Prune a Conv1D or nn.Linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    if isinstance(layer, nn.Linear):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError("Can't prune layer of class {}".format(layer.__class__))


def apply_chunking_to_forward(
    chunk_size: int, chunk_dim: int, forward_fn: Callable[..., torch.Tensor], *input_tensors
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension `chunk_dim`.
    It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the
    same result as not applying it.

    Args:
        chunk_size: int - the chunk size of a chunked tensor. `num_chunks` = `len(input_tensors[0]) / chunk_size`
        chunk_dim: int - the dimension over which the input_tensors should be chunked
        forward_fn: fn - the forward fn of the model
        input_tensors: tuple(torch.Tensor) - the input tensors of `forward_fn` which are chunked
    Returns:
        a Tensor with the same shape the foward_fn would have given if applied


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.chunk_size_lm_head, self.seq_len_dim, self.forward_chunk, hidden_states)
    """

    assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
    tensor_shape = input_tensors[0].shape
    assert all(
        input_tensor.shape == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compability
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    assert num_args_in_forward_chunk_fn == len(
        input_tensors
    ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
        num_args_in_forward_chunk_fn, len(input_tensors)
    )

    if chunk_size > 0:
        assert (
            input_tensors[0].shape[chunk_dim] % chunk_size == 0
        ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
            input_tensors[0][chunk_dim], chunk_size
        )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for python and fast tokenizers. Fast tokenizers are provided by HuggingFace's tokenizers library."""

import copy
import functools
import itertools
import json
import logging
import operator
import os
import re
from collections import UserDict, defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from tokenizers import AddedToken as AddedTokenFast
from tokenizers import Encoding as EncodingFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.implementations import BaseTokenizer as BaseTokenizerFast

from .file_utils import cached_path, hf_bucket_url, is_remote_url, is_tf_available, is_torch_available, torch_required


if is_tf_available():
    import tensorflow as tf
if is_torch_available():
    import torch

logger = logging.getLogger(__name__)

SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


class CharSpan(NamedTuple):
    """ Character span in the original string

        Args:
            start: index of the first character in the original string
            end: index of the character following the last character in the original string
    """

    start: int
    end: int


class TokenSpan(NamedTuple):
    """ Token span in an encoded string (list of tokens)

        Args:
            start: index of the first token in the span
            end: index of the token following the last token in the span
    """

    start: int
    end: int


def flatten(x: Sequence):
    """
    Flatten the provided (potentially nested) sequence

    Args:
        x (Sequence): Potentially nested sequence to flatten

    Returns:
        list: Flattened sequence
    """

    return functools.reduce(operator.iconcat, x, [])


@contextmanager
def truncate_and_pad(
    tokenizer: BaseTokenizerFast,
    max_length: int,
    stride: int,
    strategy: str,
    pad_to_max_length: bool,
    padding_side: str,
    pad_token_id: int,
    pad_token_type_id: int,
    pad_token: str,
):
    """ This contextmanager is in charge of defining the truncation and the padding strategies for fast tokenizers
        (provided by HuggingFace tokenizers library) and restore the tokenizer settings afterwards.

        This contextmanager assumes the provider tokenizer has no padding / truncation strategy
        before the managed section. If your tokenizer set a padding / truncation strategy before,
        then it will be reset to no padding/truncation when exiting the managed section.

        Args:
            tokenizer (BaseTokenizerFast): The tokenizer which will be used
            max_length (int): The maximum size of the sequence
            stride (int): The stride to use when handling overflow
            strategy (str): Overflowing logic to use
            pad_to_max_length (bool): Boolean indicating if the output needs to be padded up to max_length
            padding_side (str): "left" or "right" indicating the direction the output sequence will be padded
            pad_token_id (int): The integer representation of the padding token to use
            pad_token_type_id (int): The integer representation of the padding token type to use
            pad_token (str): The string representation of the padding token to use

    """

    # Handle all the truncation and padding stuff
    if max_length is not None:
        tokenizer.enable_truncation(max_length, stride=stride, strategy=strategy)

    if pad_to_max_length and (pad_token and pad_token_id >= 0):
        tokenizer.enable_padding(
            max_length=max_length,
            direction=padding_side,
            pad_id=pad_token_id,
            pad_type_id=pad_token_type_id,
            pad_token=pad_token,
        )
    elif pad_to_max_length:
        logger.warning(
            "Disabled padding because no padding token set (pad_token: {}, pad_token_id: {}).\n"
            "To remove this error, you can add a new pad token and then resize model embedding:\n"
            "\ttokenizer.pad_token = '<PAD>'\n\tmodel.resize_token_embeddings(len(tokenizer))".format(
                pad_token, pad_token_id
            )
        )

    yield

    # TODO(morgan, anthony): once we have a simple way to serialize tokenizers maybe store and restore the state afterward
    # to avoid destructing the padding / truncation strategy as we do now.

    if max_length is not None:
        tokenizer.no_truncation()

    if pad_to_max_length and (pad_token and pad_token_id >= 0):
        tokenizer.no_padding()


class BatchEncoding(UserDict):
    """ BatchEncoding hold the output of the encode and batch_encode methods (tokens, attention_masks, etc).
        This class is derived from a python Dictionary and can be used as a dictionnary.
        In addition, this class expose utility methods to map from word/char space to token space.

        Args:
            data (:obj:`dict`): Dictionary of lists/arrays returned by the encode/batch_encode methods ('input_ids', 'attention_mask'...)
            encoding (:obj:`EncodingFast`, :obj:`list(EncodingFast)`, `optional`, defaults to :obj:`None`):
                If the tokenizer is a fast tokenizer which outputs additional informations like mapping from word/char space to token space
                the `EncodingFast` instance or list of instance (for batches) hold these informations.

    """

    def __init__(self, data: Dict[str, Any], encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None):
        super().__init__(data)

        if isinstance(encoding, EncodingFast):
            encoding = [encoding]

        self._encodings = encoding

    def __getitem__(self, item: Union[int, str]) -> EncodingFast:
        """ If the key is a string, get the value of the dict associated to `key` ('input_ids', 'attention_mask'...)
            If the key is an integer, get the EncodingFast for batch item with index `key`
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        else:
            raise KeyError(
                "Indexing with integers (to access backend Encoding for a given batch index) "
                "is not available when using Python based tokenizers"
            )

    def __getattr__(self, item: str):
        return self.data[item]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    # After this point:
    # Extended properties and methods only available for fast (Rust-based) tokenizers
    # provided by HuggingFace tokenizers library.

    @property
    def encodings(self) -> Optional[List[EncodingFast]]:
        """
        Return the list all encoding from the tokenization process

        Returns: List[EncodingFast] or None if input was tokenized through Python (i.e. not fast) tokenizer
        """
        return self._encodings

    def tokens(self, batch_index: int = 0) -> List[int]:
        if not self._encodings:
            raise ValueError("tokens() is not available when using Python based tokenizers")
        return self._encodings[batch_index].tokens

    def words(self, batch_index: int = 0) -> List[Optional[int]]:
        if not self._encodings:
            raise ValueError("words() is not available when using Python based tokenizers")
        return self._encodings[batch_index].words

    def token_to_word(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        """ Get the index of the word corresponding (i.e. comprising) to an encoded token
            in a sequence of the batch.

            Can be called as:
                - self.token_to_word(token_index) if batch size is 1
                - self.token_to_word(batch_index, token_index) if batch size is greater than 1

            This method is particularly suited when the input sequences are provided as
            pre-tokenized sequences (i.e. words are defined by the user). In this case it allows
            to easily associate encoded tokens with provided tokenized words.

        Args:
            batch_or_token_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence,
                this can be the index of the token in the sequence
            token_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the token in the sequence.

        Returns:
            word_index (:obj:`int`):
                index of the word in the input sequence.

        """

        if not self._encodings:
            raise ValueError("token_to_word() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_word(token_index)

    def word_to_tokens(self, batch_or_word_index: int, word_index: Optional[int] = None) -> TokenSpan:
        """ Get the encoded token span corresponding to a word in the sequence of the batch.

            Token spans are returned as a TokenSpan NamedTuple with:
                start: index of the first token
                end: index of the token following the last token

            Can be called as:
                - self.word_to_tokens(word_index) if batch size is 1
                - self.word_to_tokens(batch_index, word_index) if batch size is greater or equal to 1

            This method is particularly suited when the input sequences are provided as
            pre-tokenized sequences (i.e. words are defined by the user). In this case it allows
            to easily associate encoded tokens with provided tokenized words.

        Args:
            batch_or_word_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence,
                this can be the index of the word in the sequence
            word_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the word in the sequence.

        Returns:
            token_span (:obj:`TokenSpan`):
                Span of tokens in the encoded sequence.

                TokenSpan are NamedTuple with:
                    start: index of the first token
                    end: index of the token following the last token
        """

        if not self._encodings:
            raise ValueError("word_to_tokens() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if word_index < 0:
            word_index = self._seq_len + word_index
        return TokenSpan(*(self._encodings[batch_index].word_to_tokens(word_index)))

    def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int] = None) -> CharSpan:
        """ Get the character span corresponding to an encoded token in a sequence of the batch.

            Character spans are returned as a CharSpan NamedTuple with:
                start: index of the first character in the original string associated to the token
                end: index of the character following the last character in the original string associated to the token

            Can be called as:
                - self.token_to_chars(token_index) if batch size is 1
                - self.token_to_chars(batch_index, token_index) if batch size is greater or equal to 1

        Args:
            batch_or_token_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence,
                this can be the index of the token in the sequence
            token_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the token or tokens in the sequence.

        Returns:
            char_span (:obj:`CharSpan`):
                Span of characters in the original string.

                CharSpan are NamedTuple with:
                    start: index of the first character in the original string
                    end: index of the character following the last character in the original string
        """

        if not self._encodings:
            raise ValueError("token_to_chars() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        return CharSpan(*(self._encodings[batch_index].token_to_chars(token_index)))

    def char_to_token(self, batch_or_char_index: int, char_index: Optional[int] = None) -> int:
        """ Get the index of the token in the encoded output comprising a character
            in the original string for a sequence of the batch.

            Can be called as:
                - self.char_to_token(char_index) if batch size is 1
                - self.char_to_token(batch_index, char_index) if batch size is greater or equal to 1

            This method is particularly suited when the input sequences are provided as
            pre-tokenized sequences (i.e. words are defined by the user). In this case it allows
            to easily associate encoded tokens with provided tokenized words.

        Args:
            batch_or_char_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence,
                this can be the index of the word in the sequence
            char_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the word in the sequence.


        Returns:
            token_index (:obj:`int`):
                Index of the token.
        """

        if not self._encodings:
            raise ValueError("char_to_token() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_token(char_index)

    def word_to_chars(self, batch_or_word_index: int, word_index: Optional[int] = None) -> CharSpan:
        """ Get the character span in the original string corresponding to given word in a sequence
            of the batch.

            Character spans are returned as a CharSpan NamedTuple with:
                start: index of the first character in the original string
                end: index of the character following the last character in the original string

            Can be called as:
                - self.word_to_chars(word_index) if batch size is 1
                - self.word_to_chars(batch_index, word_index) if batch size is greater or equal to 1

        Args:
            batch_or_word_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence,
                this can be the index of the word in the sequence
            word_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the word in the sequence.

        Returns:
            char_span (:obj:`CharSpan` or :obj:`List[CharSpan]`):
                Span(s) of the associated character or characters in the string.
                CharSpan are NamedTuple with:
                    start: index of the first character associated to the token in the original string
                    end: index of the character following the last character associated to the token in the original string
        """

        if not self._encodings:
            raise ValueError("word_to_chars() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        return CharSpan(*(self._encodings[batch_index].word_to_chars(word_index)))

    def char_to_word(self, batch_or_char_index: int, char_index: Optional[int] = None) -> int:
        """ Get the word in the original string corresponding to a character in the original string of
            a sequence of the batch.

            Can be called as:
                - self.char_to_word(char_index) if batch size is 1
                - self.char_to_word(batch_index, char_index) if batch size is greater than 1

            This method is particularly suited when the input sequences are provided as
            pre-tokenized sequences (i.e. words are defined by the user). In this case it allows
            to easily associate encoded tokens with provided tokenized words.

        Args:
            batch_or_char_index (:obj:`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence,
                this can be the index of the character in the orginal string.
            char_index (:obj:`int`, `optional`):
                If a batch index is provided in `batch_or_token_index`, this can be the index
                of the character in the orginal string.


        Returns:
            token_index (:obj:`int` or :obj:`List[int]`):
                Index or indices of the associated encoded token(s).
        """

        if not self._encodings:
            raise ValueError("char_to_word() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_word(char_index)

    @torch_required
    def to(self, device: str):
        """Send all values to device by calling v.to(device)"""
        self.data = {k: v.to(device) for k, v in self.data.items()}
        return self


class SpecialTokensMixin:
    """ SpecialTokensMixin is derived by ``PreTrainedTokenizer`` and ``PreTrainedTokenizerFast`` and
        handles specific behaviors related to special tokens. In particular, this class hold the
        attributes which can be used to directly access to these special tokens in a
        model-independant manner and allow to set and update the special tokens.
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(self, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)
                elif isinstance(value, AddedTokenFast):
                    setattr(self, key, str(value))
                elif isinstance(value, str):
                    setattr(self, key, value)
                else:
                    raise TypeError(
                        "special token {} has to be either str or AddedTokenFast but got: {}".format(key, type(value))
                    )

    @property
    def bos_token(self):
        """ Beginning of sentence token (string). Log an error if used while not having been set. """
        if self._bos_token is None:
            logger.error("Using bos_token, but it is not set yet.")
        return self._bos_token

    @property
    def eos_token(self):
        """ End of sentence token (string). Log an error if used while not having been set. """
        if self._eos_token is None:
            logger.error("Using eos_token, but it is not set yet.")
        return self._eos_token

    @property
    def unk_token(self):
        """ Unknown token (string). Log an error if used while not having been set. """
        if self._unk_token is None:
            logger.error("Using unk_token, but it is not set yet.")
        return self._unk_token

    @property
    def sep_token(self):
        """ Separation token (string). E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
        if self._sep_token is None:
            logger.error("Using sep_token, but it is not set yet.")
        return self._sep_token

    @property
    def pad_token(self):
        """ Padding token (string). Log an error if used while not having been set. """
        if self._pad_token is None:
            logger.error("Using pad_token, but it is not set yet.")
        return self._pad_token

    @property
    def cls_token(self):
        """ Classification token (string). E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        if self._cls_token is None:
            logger.error("Using cls_token, but it is not set yet.")
        return self._cls_token

    @property
    def mask_token(self):
        """ Mask token (string). E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
        if self._mask_token is None:
            logger.error("Using mask_token, but it is not set yet.")
        return self._mask_token

    @property
    def additional_special_tokens(self):
        """ All the additional special tokens you may want to use (list of strings). Log an error if used while not having been set. """
        if self._additional_special_tokens is None:
            logger.error("Using additional_special_tokens, but it is not set yet.")
        return self._additional_special_tokens

    def _maybe_update_backend(self, value):
        """ To be overriden by derived class if a backend tokenizer has to be updated. """
        pass

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value
        self._maybe_update_backend([value])

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value
        self._maybe_update_backend([value])

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value
        self._maybe_update_backend([value])

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value
        self._maybe_update_backend([value])

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self._maybe_update_backend([value])

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value
        self._maybe_update_backend([value])

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value
        self._maybe_update_backend([value])

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value
        self._maybe_update_backend(value)

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self):
        """ Id of the unknown token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self):
        """ Id of the separation token in the vocabulary. E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self):
        """ Id of the padding token in the vocabulary. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self):
        """ Id of the padding token type in the vocabulary."""
        return self._pad_token_type_id

    @property
    def cls_token_id(self):
        """ Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self):
        """ Id of the mask token in the vocabulary. E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers). Log an error if used while not having been set. """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @property
    def special_tokens_map(self):
        """ A dictionary mapping special token class attribute (cls_token, unk_token...) to their
            values ('<unk>', '<cls>'...)
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self):
        """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            (cls_token, unk_token...).
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        """ List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
            class attributes (cls_token, unk_token...).
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids


class PreTrainedTokenizer(SpecialTokensMixin):
    """ Base class for all tokenizers.

    Handle all the shared methods for tokenization and special tokens as well as methods
    downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't
    have to handle the specific vocabulary augmentation methods of the various underlying
    dictionary structures (BPE, sentencepiece...).

    Class attributes (overridden by derived classes):

        - ``vocab_files_names``: a python ``dict`` with, as keys, the ``__init__`` keyword name of each vocabulary file
            required by the model, and as associated values, the filename for saving the associated file (string).
        - ``pretrained_vocab_files_map``: a python ``dict of dict`` the high-level keys
            being the ``__init__`` keyword name of each vocabulary file required by the model, the low-level being the
            `short-cut-names` (string) of the pretrained models with, as associated values, the `url` (string) to the
            associated pretrained vocabulary file.
        - ``max_model_input_sizes``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the pretrained
            models, and as associated values, the maximum length of the sequence inputs of this model, or None if the
            model has no maximum input size.
        - ``pretrained_init_configuration``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the
            pretrained models, and as associated values, a dictionnary of specific arguments to pass to the
            ``__init__``method of the tokenizer class for this pretrained model when loading the tokenizer with the
            ``from_pretrained()`` method.

    Args:
        - ``model_max_length``: (`Optional`) int: the maximum length in number of tokens for the inputs to the transformer model.
            When the tokenizer is loaded with `from_pretrained`, this will be set to the value stored for the associated
            model in ``max_model_input_sizes`` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).
            no associated max_length can be found in ``max_model_input_sizes``.
        - ``padding_side``: (`Optional`) string: the side on which the model should have padding applied.
            Should be selected between ['right', 'left']
        - ``model_input_names``: (`Optional`) List[string]: the list of the forward pass inputs accepted by the
            model ("token_type_ids", "attention_mask"...).
        - ``bos_token``: (`Optional`) string: a beginning of sentence token.
            Will be associated to ``self.bos_token`` and ``self.bos_token_id``
        - ``eos_token``: (`Optional`) string: an end of sentence token.
            Will be associated to ``self.eos_token`` and ``self.eos_token_id``
        - ``unk_token``: (`Optional`) string: an unknown token.
            Will be associated to ``self.unk_token`` and ``self.unk_token_id``
        - ``sep_token``: (`Optional`) string: a separation token (e.g. to separate context and query in an input sequence).
            Will be associated to ``self.sep_token`` and ``self.sep_token_id``
        - ``pad_token``: (`Optional`) string: a padding token.
            Will be associated to ``self.pad_token`` and ``self.pad_token_id``
        - ``cls_token``: (`Optional`) string: a classification token (e.g. to extract a summary of an input sequence
            leveraging self-attention along the full depth of the model).
            Will be associated to ``self.cls_token`` and ``self.cls_token_id``
        - ``mask_token``: (`Optional`) string: a masking token (e.g. when training a model with masked-language
            modeling). Will be associated to ``self.mask_token`` and ``self.mask_token_id``
        - ``additional_special_tokens``: (`Optional`) list: a list of additional special tokens.
            Adding all special tokens here ensure they won't be split by the tokenization process.
            Will be associated to ``self.additional_special_tokens`` and ``self.additional_special_tokens_ids``
    """

    vocab_files_names: Dict[str, str] = {}
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}
    max_model_input_sizes: Dict[str, int] = {}
    model_input_names: List[str] = ["token_type_ids", "attention_mask"]

    padding_side: str = "right"

    NO_PAD_TOKEN_FOR_BATCH_MSG = (
        "No padding token is set for this model, therefore no batch can be made with uneven "
        "sequences. Set a padding token or adjust the lengths of the sequences building the "
        "batch so that every sequence is of the same length."
    )

    UNEVEN_SEQUENCES_FOR_BATCH_MSG = (
        "The sequences building the batch are not of the same size, no tensor "
        "can be built. Set `pad_to_max_length=True` to pad the smaller sequences"
        "up to the larger sequence's length."
    )

    @property
    def vocab_size(self) -> int:
        """ Size of the base vocabulary (without the added tokens) """
        raise NotImplementedError

    @property
    def is_fast(self):
        return False

    @property
    def max_len(self):
        """ Kept here for backward compatibility.
            Now renamed to `model_max_length` to avoid ambiguity.
        """
        return self.model_max_length

    @property
    def max_len_single_sentence(self):
        return self.model_max_length - self.num_special_tokens_to_add(pair=False)

    @property
    def max_len_sentences_pair(self):
        return self.model_max_length - self.num_special_tokens_to_add(pair=True)

    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value):
        """ For backward compatibility, allow to try to setup 'max_len_single_sentence' """
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=False):
            logger.warning(
                "Setting 'max_len_single_sentence' is now deprecated. " "This value is automatically set up."
            )
        else:
            raise ValueError(
                "Setting 'max_len_single_sentence' is now deprecated. " "This value is automatically set up."
            )

    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value):
        """ For backward compatibility, allow to try to setup 'max_len_sentences_pair' """
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=True):
            logger.warning(
                "Setting 'max_len_sentences_pair' is now deprecated. " "This value is automatically set up."
            )
        else:
            raise ValueError(
                "Setting 'max_len_sentences_pair' is now deprecated. " "This value is automatically set up."
            )

    def get_vocab(self):
        """ Returns the vocabulary as a dict of {token: index} pairs. `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the vocab. """
        raise NotImplementedError()

    def __init__(self, model_max_length=None, **kwargs):

        super().__init__(**kwargs)

        # For backward compatibility we fallback to set model_max_length from max_len if provided
        model_max_length = model_max_length if model_max_length is not None else kwargs.pop("max_len", None)
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER

        # Padding side is right by default and overridden in subclasses. If specified in the kwargs, it is changed.
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        assert self.padding_side in [
            "right",
            "left",
        ], f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
        self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)

        # Added tokens
        self.added_tokens_encoder = {}
        self.unique_added_tokens_encoder = set()
        self.added_tokens_decoder = {}

        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {}

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return self.vocab_size + len(self.added_tokens_encoder)

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        r"""
        Instantiate a :class:`~transformers.PreTrainedTokenizer` (or a derived class) from a predefined tokenizer.

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a predefined tokenizer to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a predefined tokenizer that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing vocabulary files required by the tokenizer, for instance saved using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - (not applicable to all derived classes, deprecated) a path or url to a single saved vocabulary file if and only if the tokenizer only requires a single vocabulary file (e.g. Bert, XLNet), e.g.: ``./my_model_directory/vocab.txt``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the vocabulary files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            inputs: (`optional`) positional arguments: will be passed to the Tokenizer ``__init__`` method.

            kwargs: (`optional`) keyword arguments: will be passed to the Tokenizer ``__init__`` method. Can be used to set special tokens like ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``. See parameters in the doc string of :class:`~transformers.PreTrainedTokenizer` for details.

        Examples::

            # We can't instantiate directly the base class `PreTrainedTokenizer` so let's show our examples on a derived class: BertTokenizer

            # Download vocabulary from S3 and cache.
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Download vocabulary from S3 (user-uploaded) and cache.
            tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/')

            # If the tokenizer uses a single vocabulary file, you can point directly to this file
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/my_vocab.txt')

            # You can link tokens to special vocabulary when instantiating
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', unk_token='<unk>')
            # You should be sure '<unk>' is in the vocabulary when doing that.
            # Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
            assert tokenizer.unk_token == '<unk>'

        """
        return cls._from_pretrained(*inputs, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        s3_models = list(cls.max_model_input_sizes.keys())
        vocab_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in s3_models:
            # Get the vocabulary from AWS S3 bucket
            for file_id, map_list in cls.pretrained_vocab_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            if (
                cls.pretrained_init_configuration
                and pretrained_model_name_or_path in cls.pretrained_init_configuration
            ):
                init_configuration = cls.pretrained_init_configuration[pretrained_model_name_or_path].copy()
        else:
            # Get the vocabulary from local files
            logger.info(
                "Model name '{}' not found in model shortcut name list ({}). "
                "Assuming '{}' is a path, a model identifier, or url to a directory containing tokenizer files.".format(
                    pretrained_model_name_or_path, ", ".join(s3_models), pretrained_model_name_or_path
                )
            )

            if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                if len(cls.vocab_files_names) > 1:
                    raise ValueError(
                        "Calling {}.from_pretrained() with the path to a single file or url is not supported."
                        "Use a model identifier or the path to a directory instead.".format(cls.__name__)
                    )
                logger.warning(
                    "Calling {}.from_pretrained() with the path to a single file or url is deprecated".format(
                        cls.__name__
                    )
                )
                file_id = list(cls.vocab_files_names.keys())[0]
                vocab_files[file_id] = pretrained_model_name_or_path
            else:
                # At this point pretrained_model_name_or_path is either a directory or a model identifier name
                additional_files_names = {
                    "added_tokens_file": ADDED_TOKENS_FILE,
                    "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,
                    "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
                }
                # Look for the tokenizer main vocabulary files + the additional tokens files
                for file_id, file_name in {**cls.vocab_files_names, **additional_files_names}.items():
                    if os.path.isdir(pretrained_model_name_or_path):
                        full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                        if not os.path.exists(full_file_name):
                            logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
                            full_file_name = None
                    else:
                        full_file_name = hf_bucket_url(
                            pretrained_model_name_or_path, filename=file_name, use_cdn=False
                        )

                    vocab_files[file_id] = full_file_name

        # Get files from url, cache, or disk depending on the case
        try:
            resolved_vocab_files = {}
            for file_id, file_path in vocab_files.items():
                if file_path is None:
                    resolved_vocab_files[file_id] = None
                else:
                    resolved_vocab_files[file_id] = cached_path(
                        file_path,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                    )
        except EnvironmentError:
            if pretrained_model_name_or_path in s3_models:
                msg = "Couldn't reach server at '{}' to download vocabulary files."
            else:
                msg = (
                    "Model name '{}' was not found in tokenizers model name list ({}). "
                    "We assumed '{}' was a path or url to a directory containing vocabulary files "
                    "named {}, but couldn't find such vocabulary files at this path or url.".format(
                        pretrained_model_name_or_path,
                        ", ".join(s3_models),
                        pretrained_model_name_or_path,
                        list(cls.vocab_files_names.values()),
                    )
                )

            raise EnvironmentError(msg)

        if all(full_file_name is None for full_file_name in resolved_vocab_files.values()):
            raise EnvironmentError(
                "Model name '{}' was not found in tokenizers model name list ({}). "
                "We assumed '{}' was a path, a model identifier, or url to a directory containing vocabulary files "
                "named {} but couldn't find such vocabulary files at this path or url.".format(
                    pretrained_model_name_or_path,
                    ", ".join(s3_models),
                    pretrained_model_name_or_path,
                    list(cls.vocab_files_names.values()),
                )
            )

        for file_id, file_path in vocab_files.items():
            if file_path == resolved_vocab_files[file_id]:
                logger.info("loading file {}".format(file_path))
            else:
                logger.info("loading file {} from cache at {}".format(file_path, resolved_vocab_files[file_id]))

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            init_kwargs = init_configuration

        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        # Set max length if needed
        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            model_max_length = cls.max_model_input_sizes[pretrained_model_name_or_path]
            if model_max_length is not None and isinstance(model_max_length, (int, float)):
                init_kwargs["model_max_length"] = min(init_kwargs.get("model_max_length", int(1e30)), model_max_length)

        # Merge resolved_vocab_files arguments in init_kwargs.
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        special_tokens_map_file = resolved_vocab_files.pop("special_tokens_map_file", None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
        if special_tokens_map_file is not None:
            with open(special_tokens_map_file, encoding="utf-8") as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
            for key, value in special_tokens_map.items():
                if key not in init_kwargs:
                    init_kwargs[key] = value

        # Instantiate tokenizer.
        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except OSError:
            raise OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )

        # Save inputs and kwargs for saving and re-loading with ``save_pretrained``
        tokenizer.init_inputs = init_inputs
        tokenizer.init_kwargs = init_kwargs

        # update unique_added_tokens_encoder with special tokens for correct tokenization
        tokenizer.unique_added_tokens_encoder.update(set(tokenizer.all_special_tokens))

        # Add supplementary tokens.
        if added_tokens_file is not None:
            with open(added_tokens_file, encoding="utf-8") as added_tokens_handle:
                added_tok_encoder = json.load(added_tokens_handle)
            added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
            tokenizer.added_tokens_encoder.update(added_tok_encoder)
            tokenizer.added_tokens_decoder.update(added_tok_decoder)
            tokenizer.unique_added_tokens_encoder.update(set(tokenizer.added_tokens_encoder.keys()))

        return tokenizer

    def save_pretrained(self, save_directory):
        """ Save the tokenizer vocabulary files together with:
                - added tokens,
                - special-tokens-to-class-attributes-mapping,
                - tokenizer instantiation positional and keywords inputs (e.g. do_lower_case for Bert).

            Warning: This won't save modifications you may have applied to the tokenizer after the instantiation
            (e.g. modifying tokenizer.do_lower_case after creation).

            This method make sure the full tokenizer can then be re-loaded using the
            :func:`~transformers.PreTrainedTokenizer.from_pretrained` class method.
        """
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
            return

        special_tokens_map_file = os.path.join(save_directory, SPECIAL_TOKENS_MAP_FILE)
        added_tokens_file = os.path.join(save_directory, ADDED_TOKENS_FILE)
        tokenizer_config_file = os.path.join(save_directory, TOKENIZER_CONFIG_FILE)

        tokenizer_config = copy.deepcopy(self.init_kwargs)
        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))

        if len(self.added_tokens_encoder) > 0:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = json.dumps(self.added_tokens_encoder, ensure_ascii=False)
                f.write(out_str)

        vocab_files = self.save_vocabulary(save_directory)

        return vocab_files + (special_tokens_map_file, added_tokens_file)

    def save_vocabulary(self, save_directory):
        """ Save the tokenizer vocabulary to a directory. This method does *NOT* save added tokens
            and special token mappings.

            Please use :func:`~transformers.PreTrainedTokenizer.save_pretrained` `()` to save the full
            Tokenizer state if you want to reload it using the :func:`~transformers.PreTrainedTokenizer.from_pretrained`
            class method.
        """
        raise NotImplementedError

    def add_tokens(self, new_tokens):
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current vocabulary.

        Args:
            new_tokens: string or list of string. Each string is a token to add. Tokens are only added if they are not
            already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        """
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, list):
            new_tokens = [new_tokens]

        to_add_tokens = []
        for token in new_tokens:
            assert isinstance(token, str)
            if self.init_kwargs.get("do_lower_case", False) and token not in self.all_special_tokens:
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in to_add_tokens
            ):
                to_add_tokens.append(token)
                logger.info("Adding %s to the vocabulary", token)

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(to_add_tokens))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.unique_added_tokens_encoder = set(self.added_tokens_encoder.keys()).union(set(self.all_special_tokens))
        self.added_tokens_decoder.update(added_tok_decoder)

        return len(to_add_tokens)

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.

        Returns:
            Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def add_special_tokens(self, special_tokens_dict):
        """
        Add a dictionary of special tokens (eos, pad, cls...) to the encoder and link them
        to class attributes. If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).

        Using `add_special_tokens` will ensure your special tokens can be used in several ways:

        - special tokens are carefully handled by the tokenizer (they are never split)
        - you can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained models (ex: BertTokenizer cls_token is already registered to be '[CLS]' and XLM's one is also registered to be '</s>')

        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to add a new classification token to GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2Model.from_pretrained('gpt2')

            special_tokens_dict = {'cls_token': '<CLS>'}

            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

            assert tokenizer.cls_token == '<CLS>'
        """
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES
            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(isinstance(t, str) for t in value)
                added_tokens += self.add_tokens(value)
            else:
                assert isinstance(value, str)
                added_tokens += self.add_tokens([value])
            logger.info("Assigning %s to the %s key of the tokenizer", value, key)
            setattr(self, key, value)

        return added_tokens

    def tokenize(self, text: TextInput, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.

            Args:
                text (:obj:`string`): The sequence to be encoded.
                **kwargs (:obj: `dict`): Arguments passed to the model-specific `prepare_for_tokenization` preprocessing method.
        """
        all_special_tokens = self.all_special_tokens
        text = self.prepare_for_tokenization(text, **kwargs)

        # TODO: should this be in the base class?
        def lowercase_text(t):
            # convert non-special tokens to lowercase
            escaped_special_toks = [re.escape(s_tok) for s_tok in all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

        if self.init_kwargs.get("do_lower_case", False):
            text = lowercase_text(text)

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_added_tokens_encoder:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token) if token not in self.unique_added_tokens_encoder else [token]
                        for token in tokenized_text
                    )
                )
            )

        added_tokens = self.unique_added_tokens_encoder
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        """ Converts a token string (or a sequence of tokens) in a single integer id
            (or a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
        """
        Converts a string in a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing ``self.convert_tokens_to_ids(self.tokenize(text))``.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length (:obj:`int`, `optional`, defaults to :obj:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary.
                You can set it to the maximal input size of the model with `max_length = tokenizer.model_max_length`.
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length. The tokenizer padding sides are handled by the class attribute `padding_side`
                which can be set to the following strings:

                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors (:obj:`str`, `optional`, defaults to :obj:`None`):
                Can be set to 'tf' or 'pt' to return respectively TensorFlow :obj:`tf.constant`
                or PyTorch :obj:`torch.Tensor` instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """
        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            stride=stride,
            truncation_strategy=truncation_strategy,
            pad_to_max_length=pad_to_max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        is_pretokenized: bool = False,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]` (the later only for not-fast tokenizers)):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length (:obj:`int`, `optional`, defaults to :obj:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
                You can set it to the maximal input size of the model with `max_length = tokenizer.model_max_length`.
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length. The tokenizer padding sides are handled by the class attribute `padding_side`
                which can be set to the following strings:

                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Set to True to indicate the input is already tokenized
            return_tensors (:obj:`str`, `optional`, defaults to :obj:`None`):
                Can be set to 'tf' or 'pt' to return respectively TensorFlow :obj:`tf.constant`
                or PyTorch :obj:`torch.Tensor` instead of a list of python integers.
            return_token_type_ids (:obj:`bool`, `optional`, defaults to :obj:`None`):
                Whether to return token type IDs. If left to the default, will return the token type IDs according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are token type IDs? <../glossary.html#token-type-ids>`_
            return_attention_mask (:obj:`bool`, `optional`, defaults to :obj:`none`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return overflowing token information (default False).
            return_special_tokens_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return special tokens mask information (default False).
            return_offsets_mapping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return (char_start, char_end) for each token (default False).
                If using Python's tokenizer, this method will raise NotImplementedError.
                This one is only available on fast tokenizers inheriting from PreTrainedTokenizerFast.
            **kwargs: passed to the `self.tokenize()` method

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    attention_mask: list[int] if return_attention_mask is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True``
                    and return_special_tokens_mask is True
                }

            With the fields:

            - ``input_ids``: list of token ids to be fed to a model
            - ``token_type_ids``: list of token type ids to be fed to a model
            - ``attention_mask``: list of indices specifying which tokens should be attended to by the model
            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
            - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
            - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
              tokens and 1 specifying sequence tokens.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, add_special_tokens=add_special_tokens, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. "
                "In this case please set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via the function add_special_tokens if you want to use a padding strategy"
            )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            add_special_tokens=add_special_tokens,
            stride=stride,
            truncation_strategy=truncation_strategy,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
        )

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        is_pretokenized: bool = False,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_masks: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_masks: bool = False,
        return_offsets_mapping: bool = False,
        return_lengths: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            batch_text_or_text_pairs (:obj:`List[str]`,  :obj:`List[Tuple[str, str]]`,
                                      :obj:`List[List[str]]`,  :obj:`List[Tuple[List[str], List[str]]]`,
                                      and for not-fast tokenizers, also:
                                      :obj:`List[List[int]]`,  :obj:`List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded.
                This can be a list of string/string-sequences/int-sequences or a list of pair of
                string/string-sequences/int-sequence (see details in encode_plus)
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length (:obj:`int`, `optional`, defaults to :obj:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length. The tokenizer padding sides are handled by the class attribute `padding_side`
                which can be set to the following strings:

                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Set to True to indicate the input is already tokenized
            return_tensors (:obj:`str`, `optional`, defaults to :obj:`None`):
                Can be set to 'tf' or 'pt' to return respectively TensorFlow :obj:`tf.constant`
                or PyTorch :obj:`torch.Tensor` instead of a list of python integers.
            return_token_type_ids (:obj:`bool`, `optional`, defaults to :obj:`None`):
                Whether to return token type IDs. If left to the default, will return the token type IDs according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are token type IDs? <../glossary.html#token-type-ids>`_
            return_attention_masks (:obj:`bool`, `optional`, defaults to :obj:`none`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return overflowing token information (default False).
            return_special_tokens_masks (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return special tokens mask information (default False).
            return_offsets_mapping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return (char_start, char_end) for each token (default False).
                If using Python's tokenizer, this method will raise NotImplementedError. This one is only available on
                Rust-based tokenizers inheriting from PreTrainedTokenizerFast.
            return_lengths (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set the resulting dictionary will include the length of each encoded inputs
            **kwargs: passed to the `self.tokenize()` method

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[List[int]],
                    token_type_ids: list[List[int]] if return_token_type_ids is True (default)
                    attention_mask: list[List[int]] if return_attention_mask is True (default)
                    overflowing_tokens: list[List[int]] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: List[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[List[int]] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }

            With the fields:

            - ``input_ids``: list of token ids to be fed to a model
            - ``token_type_ids``: list of token type ids to be fed to a model
            - ``attention_mask``: list of indices specifying which tokens should be attended to by the model
            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
            - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
            - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
              tokens and 1 specifying sequence tokens.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, add_special_tokens=add_special_tokens, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. In this case please set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via the function add_special_tokens if you want to use a padding strategy"
            )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if isinstance(ids_or_pair_ids, (list, tuple)) and len(ids_or_pair_ids) == 2 and not is_pretokenized:
                ids, pair_ids = ids_or_pair_ids
            else:
                ids, pair_ids = ids_or_pair_ids, None

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        if max_length is None and pad_to_max_length:

            def total_sequence_length(input_pairs):
                first_ids, second_ids = input_pairs
                return len(first_ids) + (
                    self.num_special_tokens_to_add()
                    if second_ids is None
                    else (len(second_ids) + self.num_special_tokens_to_add(pair=True))
                )

            max_length = max([total_sequence_length(ids) for ids in input_ids])

        batch_outputs = {}
        for first_ids, second_ids in input_ids:
            # Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by
            # the model. It adds special tokens, truncates sequences if overflowing while taking into account
            # the special tokens and manages a window stride for overflowing tokens
            outputs = self.prepare_for_model(
                first_ids,
                pair_ids=second_ids,
                max_length=max_length,
                pad_to_max_length=pad_to_max_length,
                add_special_tokens=add_special_tokens,
                stride=stride,
                truncation_strategy=truncation_strategy,
                return_attention_mask=return_attention_masks,
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_masks,
                return_lengths=return_lengths,
                return_tensors=None,  # We will convert the whole batch to tensors at the end
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        if return_tensors is not None:

            self.convert_to_tensors_(batch_outputs, return_tensors)
        return BatchEncoding(batch_outputs)

    def convert_to_tensors_(self, batch_outputs: dict, return_tensors: str) -> None:
        # Do the tensor conversion in batch
        for key, value in batch_outputs.items():
            if return_tensors == "tf" and is_tf_available():
                try:
                    batch_outputs[key] = tf.constant(value)
                except ValueError:
                    if None in [item for sequence in value for item in sequence]:
                        raise ValueError(self.NO_PAD_TOKEN_FOR_BATCH_MSG)
                    else:
                        raise ValueError(self.UNEVEN_SEQUENCES_FOR_BATCH_MSG)
            elif return_tensors == "pt" and is_torch_available():
                try:
                    batch_outputs[key] = torch.tensor(value)
                except ValueError:
                    raise ValueError(self.UNEVEN_SEQUENCES_FOR_BATCH_MSG)
                except RuntimeError:
                    if None in [item for sequence in value for item in sequence]:
                        raise ValueError(self.NO_PAD_TOKEN_FOR_BATCH_MSG)
                    else:
                        raise

            elif return_tensors is not None:
                logger.warning(
                    "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                        return_tensors
                    )
                )

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_lengths: bool = False,
    ) -> BatchEncoding:
        """ Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            max_length: maximum length of the returned list. Will truncate by taking into account the special tokens.
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            stride: window stride for overflowing tokens. Can be useful to remove edge effect when using sequential
                list of inputs. The overflowing token will contains a part of the previous window of tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                The tokenizer padding sides are handled by the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            return_token_type_ids: (optional) Set to False to avoid returning token_type_ids (default: set to model specifics).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
            return_overflowing_tokens: (optional) Set to True to return overflowing token information (default False).
            return_special_tokens_mask: (optional) Set to True to return special tokens mask information (default False).
            return_lengths (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set the resulting dictionary will include the length of each encoded inputs

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                    length: int if return_lengths is True
                }

            With the fields:
                - ``input_ids``: list of token ids to be fed to a model
                - ``token_type_ids``: list of token type ids to be fed to a model

                - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
                - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
                - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                    tokens and 1 specifying sequence tokens.
                - ``length``: this is the length of ``input_ids``
        """
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        if max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        assert max_length is None or len(encoded_inputs["input_ids"]) <= max_length
        if max_length is None and len(encoded_inputs["input_ids"]) > self.model_max_length:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model ({} > {}). Running this sequence through the model will result in "
                "indexing errors".format(len(ids), self.model_max_length)
            )

        # Padding
        needs_to_be_padded = pad_to_max_length and (
            max_length
            and len(encoded_inputs["input_ids"]) < max_length
            or max_length is None
            and len(encoded_inputs["input_ids"]) < self.model_max_length
            and self.model_max_length <= LARGE_INTEGER
        )

        if pad_to_max_length and max_length is None and self.model_max_length > LARGE_INTEGER:
            logger.warning(
                "Sequence can't be padded as no maximum length is specified and the model maximum length is too high."
            )

        if needs_to_be_padded:
            difference = (max_length if max_length is not None else self.model_max_length) - len(
                encoded_inputs["input_ids"]
            )
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + encoded_inputs["input_ids"]
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        if return_lengths:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        # Prepare model inputs as tensors if asked
        if return_tensors == "tf" and is_tf_available():
            encoded_inputs["input_ids"] = tf.constant([encoded_inputs["input_ids"]])

            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = tf.constant([encoded_inputs["token_type_ids"]])

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = tf.constant([encoded_inputs["attention_mask"]])

        elif return_tensors == "pt" and is_torch_available():
            encoded_inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])

            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = torch.tensor([encoded_inputs["token_type_ids"]])

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])
        elif return_tensors is not None:
            logger.warning(
                "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                    return_tensors
                )
            )

        return BatchEncoding(encoded_inputs)

    def prepare_for_tokenization(self, text: str, **kwargs) -> str:
        """ Performs any necessary transformations before tokenization """
        return text

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: str = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """ Truncates a sequence pair in place to the maximum length.

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to ``0``):
                number of tokens to remove using the truncation strategy
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == "longest_first":
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == "only_first":
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == "only_second":
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == "do_not_truncate":
            raise ValueError("Input sequence are too long for max_length. Please select a truncation strategy.")
        else:
            raise ValueError(
                "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
        return (ids, pair_ids, overflowing_tokens)

    def create_token_type_ids_from_sequences(self, token_ids_0: List, token_ids_1: Optional[List] = None) -> List[int]:
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(self, token_ids_0: List, token_ids_1: Optional[List] = None) -> List:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[int, List[int]]:
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str), using the vocabulary and added tokens.

            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """ Converts a sequence of tokens (string) in a single string.
            The most simple way to do it is ' '.join(self.convert_ids_to_tokens(token_ids))
            but we often want to remove sub-word tokenization artifacts at the same time.
        """
        return " ".join(self.convert_ids_to_tokens(tokens))

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids: list of tokenized input ids. Can be obtained using the `encode` or `encode_plus` methods.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separatly for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = " ".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string


class PreTrainedTokenizerFast(PreTrainedTokenizer):
    """ Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherit from PreTrainedTokenizer.

    Handle all the shared methods for tokenization and special tokens as well as methods
    downloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't
    have to handle the specific vocabulary augmentation methods of the various underlying
    dictionary structures (BPE, sentencepiece...).

    Class attributes (overridden by derived classes):

        - ``vocab_files_names``: a python ``dict`` with, as keys, the ``__init__`` keyword name of each vocabulary file
            required by the model, and as associated values, the filename for saving the associated file (string).
        - ``pretrained_vocab_files_map``: a python ``dict of dict`` the high-level keys
            being the ``__init__`` keyword name of each vocabulary file required by the model, the low-level being the
            `short-cut-names` (string) of the pretrained models with, as associated values, the `url` (string) to the
            associated pretrained vocabulary file.
        - ``max_model_input_sizes``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the pretrained
            models, and as associated values, the maximum length of the sequence inputs of this model, or None if the
            model has no maximum input size.
        - ``pretrained_init_configuration``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the
            pretrained models, and as associated values, a dictionnary of specific arguments to pass to the
            ``__init__``method of the tokenizer class for this pretrained model when loading the tokenizer with the
            ``from_pretrained()`` method.

    Args:
        - ``tokenizer`` (`BaseTokenizerFast`): A Fast tokenizer from the HuggingFace tokenizer library (in low level Rust language)
        - ``model_max_length``: (`Optional`) int: the maximum length in number of tokens for the inputs to the transformer model.
            When the tokenizer is loaded with `from_pretrained`, this will be set to the value stored for the associated
            model in ``max_model_input_sizes`` (see above). If no value is provided, will default to VERY_LARGE_INTEGER (`int(1e30)`).
            no associated max_length can be found in ``max_model_input_sizes``.
        - ``padding_side``: (`Optional`) string: the side on which the model should have padding applied.
            Should be selected between ['right', 'left']
        - ``model_input_names``: (`Optional`) List[string]: the list of the forward pass inputs accepted by the
            model ("token_type_ids", "attention_mask"...).
        - ``bos_token``: (`Optional`) string: a beginning of sentence token.
            Will be associated to ``self.bos_token`` and ``self.bos_token_id``
        - ``eos_token``: (`Optional`) string: an end of sentence token.
            Will be associated to ``self.eos_token`` and ``self.eos_token_id``
        - ``unk_token``: (`Optional`) string: an unknown token.
            Will be associated to ``self.unk_token`` and ``self.unk_token_id``
        - ``sep_token``: (`Optional`) string: a separation token (e.g. to separate context and query in an input sequence).
            Will be associated to ``self.sep_token`` and ``self.sep_token_id``
        - ``pad_token``: (`Optional`) string: a padding token.
            Will be associated to ``self.pad_token`` and ``self.pad_token_id``
        - ``cls_token``: (`Optional`) string: a classification token (e.g. to extract a summary of an input sequence
            leveraging self-attention along the full depth of the model).
            Will be associated to ``self.cls_token`` and ``self.cls_token_id``
        - ``mask_token``: (`Optional`) string: a masking token (e.g. when training a model with masked-language
            modeling). Will be associated to ``self.mask_token`` and ``self.mask_token_id``
        - ``additional_special_tokens``: (`Optional`) list: a list of additional special tokens.
            Adding all special tokens here ensure they won't be split by the tokenization process.
            Will be associated to ``self.additional_special_tokens`` and ``self.additional_special_tokens_ids``
    """

    def __init__(self, tokenizer: BaseTokenizerFast, **kwargs):
        if not isinstance(tokenizer, BaseTokenizerFast):
            raise ValueError(
                "Tokenizer should be an instance of a Tokenizer " "provided by HuggingFace tokenizers library."
            )
        self._tokenizer: BaseTokenizerFast = tokenizer

        # Initialize all the rest of the kwargs
        super().__init__(**kwargs)

    @property
    def backend_tokenizer(self) -> BaseTokenizerFast:
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        return self._tokenizer._tokenizer.decoder

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def _maybe_update_backend(self, value):
        """ Update the backend fast tokenizer.
            Override method from base class SpecialTokensMixin """
        self._tokenizer.add_special_tokens(value)

    def _convert_encoding(
        self,
        encoding: EncodingFast,
        return_tensors: Optional[bool] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
    ) -> Dict[str, Any]:
        """ Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict.

            Overflowing tokens are converted to additional examples (like batches) so the output values of
            the dict are lists (overflows) of lists (tokens).

            If return_tensors is not None, these lists of lists are converted to 2-D tensors
            for input_ids, token_type_ids and attention_mask.
            Output shape: (overflows, sequence length)
        """
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)

            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)

        if return_tensors is not None:
            for key, value in encoding_dict.items():
                if return_tensors == "tf" and is_tf_available():
                    encoding_dict[key] = tf.constant(value)
                elif return_tensors == "pt" and is_torch_available():
                    encoding_dict[key] = torch.tensor(value)
                elif return_tensors is not None:
                    logger.warning(
                        "Unable to convert output to tensors format {}, "
                        "PyTorch or TensorFlow is not available.".format(return_tensors)
                    )

        return encoding_dict

    def _convert_token_to_id_with_added_voc(self, token: int) -> str:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    def convert_tokens_to_string(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens)

    def add_tokens(self, new_tokens: List[Union[str, AddedTokenFast]]) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current vocabulary.

        Args:
            new_tokens: string or list of string or AddedTokenFast. Each string is a token to add.
            Tokens are only added if they are not already in the vocabulary. AddedTokenFast wrap a string token to let you personnalize it's behavior (Whether this token should only match against single word, whether this token should strip all potential whitespaces on the left side, Whether this token should strip all potential whitespaces on the right side...).
            See details for AddedToken in HuggingFace tokenizers library.

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        """
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        return self._tokenizer.add_tokens(new_tokens)

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        # Map special tokens to class attributes (self.pad_token...)
        num_added_tokens = super().add_special_tokens(special_tokens_dict)

        # If the backend tokenizer the only specificities of special tokens are that
        #    - they will never be processed by the model, and
        #    - they will be removed while decoding.
        # But they are not mapped to special attributes in the backend so we can just
        # send a list.
        tokens = flatten(special_tokens_dict.values())
        self._tokenizer.add_special_tokens(tokens)

        return num_added_tokens

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self._tokenizer.num_special_tokens_to_add(pair)

    def tokenize(
        self, text: TextInput, pair: Optional[TextInput] = None, add_special_tokens: bool = False
    ) -> List[str]:
        return self._tokenizer.encode(text, pair, add_special_tokens).tokens

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        is_pretokenized: bool = False,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_lengths: bool = False,
        **kwargs
    ) -> BatchEncoding:

        if not isinstance(batch_text_or_text_pairs, list):
            raise ValueError(
                "batch_text_or_text_pairs has to be a list (got {})".format(type(batch_text_or_text_pairs))
            )

        # Needed if we have to return a tensor
        pad_to_max_length = pad_to_max_length or (return_tensors is not None and len(batch_text_or_text_pairs) > 1)

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError("Unable to set proper padding strategy as the tokenizer does not have a padding token")

        # Set the truncation and padding strategy and restore the initial configuration
        with truncate_and_pad(
            tokenizer=self._tokenizer,
            max_length=max_length,
            stride=stride,
            strategy=truncation_strategy,
            pad_to_max_length=pad_to_max_length,
            padding_side=self.padding_side,
            pad_token_id=self.pad_token_id,
            pad_token_type_id=self.pad_token_type_id,
            pad_token=self._pad_token,
        ):

            # Check for the pretokenized path
            if is_pretokenized:
                encodings = []

                # Iterate over each sample (we don't know yet if they are pairs or simple input
                for i, sample in enumerate(batch_text_or_text_pairs):

                    if not isinstance(sample, (list, tuple)):
                        raise TypeError(
                            "batch_encode_plus(..., is_pretokenized=True) requires batch_text_or_text_pairs "
                            "to be either List[List[str]] or List[Tuple[List[str], List[str]]] but sample at "
                            "index {} is of type {}".format(i, type(sample))
                        )

                    # Test if we have a pair of sentences by checking the depth of nesting
                    is_pair = bool(len(sample) > 0 and isinstance(sample[0], (list, tuple)))

                    # Take care of the first sequence - we multi-thread over the words
                    encodings_text = EncodingFast.merge(
                        self._tokenizer.encode_batch(sample[0] if is_pair else sample, add_special_tokens=False),
                        growing_offsets=True,
                    )

                    # Take care of the second sequence if we have a pair
                    if is_pair:
                        encodings_pair = EncodingFast.merge(
                            self._tokenizer.encode_batch([("", s) for s in sample[1]], add_special_tokens=False),
                            growing_offsets=True,
                        )
                    else:
                        encodings_pair = None

                    # Post-process - truncate/pad and add special tokens
                    encoding = self._tokenizer.post_process(encodings_text, encodings_pair, add_special_tokens)
                    encodings.append(encoding)

            # Classical path with strings input
            else:
                # Avoid thread overhead if only one example.
                if len(batch_text_or_text_pairs) == 1:
                    if isinstance(batch_text_or_text_pairs[0], (tuple, list)):
                        encodings = self._tokenizer.encode(
                            *batch_text_or_text_pairs[0], add_special_tokens=add_special_tokens
                        )
                    else:
                        encodings = self._tokenizer.encode(
                            batch_text_or_text_pairs[0], add_special_tokens=add_special_tokens
                        )
                    encodings = [encodings]
                else:
                    encodings = self._tokenizer.encode_batch(
                        batch_text_or_text_pairs, add_special_tokens=add_special_tokens
                    )

        # Convert encoding to dict
        # `Tokens` has type: List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]]
        # with nested dimensions corresponding to batch, overflows, sequence length
        tokens = [
            self._convert_encoding(
                encoding=encoding,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
            )
            for encoding in encodings
        ]

        # Sanitize the output to have dict[list] from list[dict]
        sanitized = {}
        for key in tokens[0].keys():
            # To List[List[List[int]]] of shape (batch, overflows, sequence length)
            stack = [e for item in tokens for e in item[key]]
            if return_tensors == "tf":
                stack = tf.stack(stack, axis=0)
            elif return_tensors == "pt":
                stack = torch.stack(stack, dim=0)
            # elif not return_tensors and len(stack) == 1:
            #     stack = stack[0]

            sanitized[key] = stack

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = flatten([[i] * len(enc["input_ids"]) for i, enc in enumerate(tokens)])
            sanitized["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        return BatchEncoding(sanitized, encodings)

    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = False,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        is_pretokenized: bool = False,
        return_tensors: Optional[bool] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs
    ) -> BatchEncoding:

        # Check for pretokenized path (ie [token1, token2, ..., tokenN] -> [id1, id2, ..., idN]
        if is_pretokenized:
            if isinstance(text, list) and len(text) > 0:

                # Encode through encode_batch with sequence of only one word which will be merged after hand
                encoding = self._tokenizer.encode_batch(text, add_special_tokens=False)
                encoding = EncodingFast.merge(encoding, growing_offsets=True)

                # Let's do the same for pairs if provided
                if isinstance(text_pair, list):
                    # We prepend empty string before each word so that encoding is aware content is a pair
                    encoding_pair = self._tokenizer.encode_batch(
                        [("", p) for p in text_pair], add_special_tokens=False
                    )
                    encoding_pair = EncodingFast.merge(encoding_pair, growing_offsets=True)
                elif text_pair is None:
                    encoding_pair = None
                else:
                    raise TypeError(
                        "encode_plus(..., is_pretokenized=True) requires text and text_pair to be List[str] "
                        "but got (text={}, text_pair={})".format(type(text), type(text_pair))
                    )

                # Post process and if asked to do so, insert special tokens where needed
                encoding = self._tokenizer.post_process(encoding, encoding_pair, add_special_tokens)

                batched_output = BatchEncoding(
                    self._convert_encoding(
                        encoding,
                        return_tensors=return_tensors,
                        return_token_type_ids=return_token_type_ids,
                        return_attention_mask=return_attention_mask,
                        return_overflowing_tokens=return_overflowing_tokens,
                        return_special_tokens_mask=return_special_tokens_mask,
                        return_offsets_mapping=return_offsets_mapping,
                    ),
                    encoding,
                )
            else:
                raise TypeError(
                    "encode_plus(..., is_pretokenized=True) requires text to be List[str] "
                    "but got (text={}, text_pair={})".format(type(text), type(text_pair))
                )
        else:
            batched_input = [(text, text_pair)] if text_pair else [text]
            batched_output = self.batch_encode_plus(
                batched_input,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                stride=stride,
                truncation_strategy=truncation_strategy,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                pad_to_max_length=pad_to_max_length,
                **kwargs,
            )

        # Return tensor is None, then we can remove the leading batch axis
        if not return_tensors:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        return batched_output

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True
    ) -> str:
        text = self._tokenizer.decode(token_ids, skip_special_tokens)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory: str) -> Tuple[str]:
        if os.path.isdir(save_directory):
            files = self._tokenizer.save(save_directory)
        else:
            folder, file = os.path.split(os.path.abspath(save_directory))
            files = self._tokenizer.save(folder, name=file)

        return tuple(files)


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])# Author:  Travis Oliphant, 2002
#
# Further enhancements and tests added by numerous SciPy developers.
#
import warnings

import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal,
    assert_almost_equal, assert_array_less, assert_array_almost_equal,
    assert_, assert_allclose, assert_equal, suppress_warnings)
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy import stats
from scipy.stats.morestats import _abw_state
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr
from scipy.stats._binomtest import _binary_search_for_binom_tst

# Matplotlib is not a scipy dependency but is optionally used in probplot, so
# check if it's available
try:
    import matplotlib  # type: ignore[import]
    matplotlib.rcParams['backend'] = 'Agg'
    import matplotlib.pyplot as plt  # type: ignore[import]
    have_matplotlib = True
except Exception:
    have_matplotlib = False


# test data gear.dat from NIST for Levene and Bartlett test
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3581.htm
g1 = [1.006, 0.996, 0.998, 1.000, 0.992, 0.993, 1.002, 0.999, 0.994, 1.000]
g2 = [0.998, 1.006, 1.000, 1.002, 0.997, 0.998, 0.996, 1.000, 1.006, 0.988]
g3 = [0.991, 0.987, 0.997, 0.999, 0.995, 0.994, 1.000, 0.999, 0.996, 0.996]
g4 = [1.005, 1.002, 0.994, 1.000, 0.995, 0.994, 0.998, 0.996, 1.002, 0.996]
g5 = [0.998, 0.998, 0.982, 0.990, 1.002, 0.984, 0.996, 0.993, 0.980, 0.996]
g6 = [1.009, 1.013, 1.009, 0.997, 0.988, 1.002, 0.995, 0.998, 0.981, 0.996]
g7 = [0.990, 1.004, 0.996, 1.001, 0.998, 1.000, 1.018, 1.010, 0.996, 1.002]
g8 = [0.998, 1.000, 1.006, 1.000, 1.002, 0.996, 0.998, 0.996, 1.002, 1.006]
g9 = [1.002, 0.998, 0.996, 0.995, 0.996, 1.004, 1.004, 0.998, 0.999, 0.991]
g10 = [0.991, 0.995, 0.984, 0.994, 0.997, 0.997, 0.991, 0.998, 1.004, 0.997]


class TestBayes_mvs:
    def test_basic(self):
        # Expected values in this test simply taken from the function.  For
        # some checks regarding correctness of implementation, see review in
        # gh-674
        data = [6, 9, 12, 7, 8, 8, 13]
        mean, var, std = stats.bayes_mvs(data)
        assert_almost_equal(mean.statistic, 9.0)
        assert_allclose(mean.minmax, (7.1036502226125329, 10.896349777387467),
                        rtol=1e-14)

        assert_almost_equal(var.statistic, 10.0)
        assert_allclose(var.minmax, (3.1767242068607087, 24.45910381334018),
                        rtol=1e-09)

        assert_almost_equal(std.statistic, 2.9724954732045084, decimal=14)
        assert_allclose(std.minmax, (1.7823367265645145, 4.9456146050146312),
                        rtol=1e-14)

    def test_empty_input(self):
        assert_raises(ValueError, stats.bayes_mvs, [])

    def test_result_attributes(self):
        x = np.arange(15)
        attributes = ('statistic', 'minmax')
        res = stats.bayes_mvs(x)

        for i in res:
            check_named_results(i, attributes)


class TestMvsdist:
    def test_basic(self):
        data = [6, 9, 12, 7, 8, 8, 13]
        mean, var, std = stats.mvsdist(data)
        assert_almost_equal(mean.mean(), 9.0)
        assert_allclose(mean.interval(0.9), (7.1036502226125329,
                                             10.896349777387467), rtol=1e-14)

        assert_almost_equal(var.mean(), 10.0)
        assert_allclose(var.interval(0.9), (3.1767242068607087,
                                            24.45910381334018), rtol=1e-09)

        assert_almost_equal(std.mean(), 2.9724954732045084, decimal=14)
        assert_allclose(std.interval(0.9), (1.7823367265645145,
                                            4.9456146050146312), rtol=1e-14)

    def test_empty_input(self):
        assert_raises(ValueError, stats.mvsdist, [])

    def test_bad_arg(self):
        # Raise ValueError if fewer than two data points are given.
        data = [1]
        assert_raises(ValueError, stats.mvsdist, data)

    def test_warns(self):
        # regression test for gh-5270
        # make sure there are no spurious divide-by-zero warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            [x.mean() for x in stats.mvsdist([1, 2, 3])]
            [x.mean() for x in stats.mvsdist([1, 2, 3, 4, 5])]


class TestShapiro:
    def test_basic(self):
        x1 = [0.11, 7.87, 4.61, 10.14, 7.95, 3.14, 0.46,
              4.43, 0.21, 4.75, 0.71, 1.52, 3.24,
              0.93, 0.42, 4.97, 9.53, 4.55, 0.47, 6.66]
        w, pw = stats.shapiro(x1)
        shapiro_test = stats.shapiro(x1)
        assert_almost_equal(w, 0.90047299861907959, decimal=6)
        assert_almost_equal(shapiro_test.statistic, 0.90047299861907959, decimal=6)
        assert_almost_equal(pw, 0.042089745402336121, decimal=6)
        assert_almost_equal(shapiro_test.pvalue, 0.042089745402336121, decimal=6)

        x2 = [1.36, 1.14, 2.92, 2.55, 1.46, 1.06, 5.27, -1.11,
              3.48, 1.10, 0.88, -0.51, 1.46, 0.52, 6.20, 1.69,
              0.08, 3.67, 2.81, 3.49]
        w, pw = stats.shapiro(x2)
        shapiro_test = stats.shapiro(x2)
        assert_almost_equal(w, 0.9590270, decimal=6)
        assert_almost_equal(shapiro_test.statistic, 0.9590270, decimal=6)
        assert_almost_equal(pw, 0.52460, decimal=3)
        assert_almost_equal(shapiro_test.pvalue, 0.52460, decimal=3)

        # Verified against R
        x3 = stats.norm.rvs(loc=5, scale=3, size=100, random_state=12345678)
        w, pw = stats.shapiro(x3)
        shapiro_test = stats.shapiro(x3)
        assert_almost_equal(w, 0.9772805571556091, decimal=6)
        assert_almost_equal(shapiro_test.statistic, 0.9772805571556091, decimal=6)
        assert_almost_equal(pw, 0.08144091814756393, decimal=3)
        assert_almost_equal(shapiro_test.pvalue, 0.08144091814756393, decimal=3)

        # Extracted from original paper
        x4 = [0.139, 0.157, 0.175, 0.256, 0.344, 0.413, 0.503, 0.577, 0.614,
              0.655, 0.954, 1.392, 1.557, 1.648, 1.690, 1.994, 2.174, 2.206,
              3.245, 3.510, 3.571, 4.354, 4.980, 6.084, 8.351]
        W_expected = 0.83467
        p_expected = 0.000914
        w, pw = stats.shapiro(x4)
        shapiro_test = stats.shapiro(x4)
        assert_almost_equal(w, W_expected, decimal=4)
        assert_almost_equal(shapiro_test.statistic, W_expected, decimal=4)
        assert_almost_equal(pw, p_expected, decimal=5)
        assert_almost_equal(shapiro_test.pvalue, p_expected, decimal=5)

    def test_2d(self):
        x1 = [[0.11, 7.87, 4.61, 10.14, 7.95, 3.14, 0.46,
              4.43, 0.21, 4.75], [0.71, 1.52, 3.24,
              0.93, 0.42, 4.97, 9.53, 4.55, 0.47, 6.66]]
        w, pw = stats.shapiro(x1)
        shapiro_test = stats.shapiro(x1)
        assert_almost_equal(w, 0.90047299861907959, decimal=6)
        assert_almost_equal(shapiro_test.statistic, 0.90047299861907959, decimal=6)
        assert_almost_equal(pw, 0.042089745402336121, decimal=6)
        assert_almost_equal(shapiro_test.pvalue, 0.042089745402336121, decimal=6)

        x2 = [[1.36, 1.14, 2.92, 2.55, 1.46, 1.06, 5.27, -1.11,
              3.48, 1.10], [0.88, -0.51, 1.46, 0.52, 6.20, 1.69,
              0.08, 3.67, 2.81, 3.49]]
        w, pw = stats.shapiro(x2)
        shapiro_test = stats.shapiro(x2)
        assert_almost_equal(w, 0.9590270, decimal=6)
        assert_almost_equal(shapiro_test.statistic, 0.9590270, decimal=6)
        assert_almost_equal(pw, 0.52460, decimal=3)
        assert_almost_equal(shapiro_test.pvalue, 0.52460, decimal=3)

    def test_empty_input(self):
        assert_raises(ValueError, stats.shapiro, [])
        assert_raises(ValueError, stats.shapiro, [[], [], []])

    def test_not_enough_values(self):
        assert_raises(ValueError, stats.shapiro, [1, 2])
        assert_raises(ValueError, stats.shapiro, np.array([[], [2]], dtype=object))

    def test_bad_arg(self):
        # Length of x is less than 3.
        x = [1]
        assert_raises(ValueError, stats.shapiro, x)

    def test_nan_input(self):
        x = np.arange(10.)
        x[9] = np.nan

        w, pw = stats.shapiro(x)
        shapiro_test = stats.shapiro(x)
        assert_equal(w, np.nan)
        assert_equal(shapiro_test.statistic, np.nan)
        assert_almost_equal(pw, 1.0)
        assert_almost_equal(shapiro_test.pvalue, 1.0)


class TestAnderson:
    def test_normal(self):
        rs = RandomState(1234567890)
        x1 = rs.standard_exponential(size=50)
        x2 = rs.standard_normal(size=50)
        A, crit, sig = stats.anderson(x1)
        assert_array_less(crit[:-1], A)
        A, crit, sig = stats.anderson(x2)
        assert_array_less(A, crit[-2:])

        v = np.ones(10)
        v[0] = 0
        A, crit, sig = stats.anderson(v)
        # The expected statistic 3.208057 was computed independently of scipy.
        # For example, in R:
        #   > library(nortest)
        #   > v <- rep(1, 10)
        #   > v[1] <- 0
        #   > result <- ad.test(v)
        #   > result$statistic
        #          A
        #   3.208057
        assert_allclose(A, 3.208057)

    def test_expon(self):
        rs = RandomState(1234567890)
        x1 = rs.standard_exponential(size=50)
        x2 = rs.standard_normal(size=50)
        A, crit, sig = stats.anderson(x1, 'expon')
        assert_array_less(A, crit[-2:])
        with np.errstate(all='ignore'):
            A, crit, sig = stats.anderson(x2, 'expon')
        assert_(A > crit[-1])

    def test_gumbel(self):
        # Regression test for gh-6306.  Before that issue was fixed,
        # this case would return a2=inf.
        v = np.ones(100)
        v[0] = 0.0
        a2, crit, sig = stats.anderson(v, 'gumbel')
        # A brief reimplementation of the calculation of the statistic.
        n = len(v)
        xbar, s = stats.gumbel_l.fit(v)
        logcdf = stats.gumbel_l.logcdf(v, xbar, s)
        logsf = stats.gumbel_l.logsf(v, xbar, s)
        i = np.arange(1, n+1)
        expected_a2 = -n - np.mean((2*i - 1) * (logcdf + logsf[::-1]))

        assert_allclose(a2, expected_a2)

    def test_bad_arg(self):
        assert_raises(ValueError, stats.anderson, [1], dist='plate_of_shrimp')

    def test_result_attributes(self):
        rs = RandomState(1234567890)
        x = rs.standard_exponential(size=50)
        res = stats.anderson(x)
        attributes = ('statistic', 'critical_values', 'significance_level')
        check_named_results(res, attributes)

    def test_gumbel_l(self):
        # gh-2592, gh-6337
        # Adds support to 'gumbel_r' and 'gumbel_l' as valid inputs for dist.
        rs = RandomState(1234567890)
        x = rs.gumbel(size=100)
        A1, crit1, sig1 = stats.anderson(x, 'gumbel')
        A2, crit2, sig2 = stats.anderson(x, 'gumbel_l')

        assert_allclose(A2, A1)

    def test_gumbel_r(self):
        # gh-2592, gh-6337
        # Adds support to 'gumbel_r' and 'gumbel_l' as valid inputs for dist.
        rs = RandomState(1234567890)
        x1 = rs.gumbel(size=100)
        x2 = np.ones(100)
        # A constant array is a degenerate case and breaks gumbel_r.fit, so
        # change one value in x2.
        x2[0] = 0.996
        A1, crit1, sig1 = stats.anderson(x1, 'gumbel_r')
        A2, crit2, sig2 = stats.anderson(x2, 'gumbel_r')

        assert_array_less(A1, crit1[-2:])
        assert_(A2 > crit2[-1])


class TestAndersonKSamp:
    def test_example1a(self):
        # Example data from Scholz & Stephens (1987), originally
        # published in Lehmann (1995, Nonparametrics, Statistical
        # Methods Based on Ranks, p. 309)
        # Pass a mixture of lists and arrays
        t1 = [38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0]
        t2 = np.array([39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8])
        t3 = np.array([34.0, 35.0, 39.0, 40.0, 43.0, 43.0, 44.0, 45.0])
        t4 = np.array([34.0, 34.8, 34.8, 35.4, 37.2, 37.8, 41.2, 42.8])

        Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4), midrank=False)

        assert_almost_equal(Tk, 4.449, 3)
        assert_array_almost_equal([0.4985, 1.3237, 1.9158, 2.4930, 3.2459],
                                  tm[0:5], 4)
        assert_allclose(p, 0.0021, atol=0.00025)

    def test_example1b(self):
        # Example data from Scholz & Stephens (1987), originally
        # published in Lehmann (1995, Nonparametrics, Statistical
        # Methods Based on Ranks, p. 309)
        # Pass arrays
        t1 = np.array([38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0])
        t2 = np.array([39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8])
        t3 = np.array([34.0, 35.0, 39.0, 40.0, 43.0, 43.0, 44.0, 45.0])
        t4 = np.array([34.0, 34.8, 34.8, 35.4, 37.2, 37.8, 41.2, 42.8])
        Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4), midrank=True)

        assert_almost_equal(Tk, 4.480, 3)
        assert_array_almost_equal([0.4985, 1.3237, 1.9158, 2.4930, 3.2459],
                                  tm[0:5], 4)
        assert_allclose(p, 0.0020, atol=0.00025)

    def test_example2a(self):
        # Example data taken from an earlier technical report of
        # Scholz and Stephens
        # Pass lists instead of arrays
        t1 = [194, 15, 41, 29, 33, 181]
        t2 = [413, 14, 58, 37, 100, 65, 9, 169, 447, 184, 36, 201, 118]
        t3 = [34, 31, 18, 18, 67, 57, 62, 7, 22, 34]
        t4 = [90, 10, 60, 186, 61, 49, 14, 24, 56, 20, 79, 84, 44, 59, 29,
              118, 25, 156, 310, 76, 26, 44, 23, 62]
        t5 = [130, 208, 70, 101, 208]
        t6 = [74, 57, 48, 29, 502, 12, 70, 21, 29, 386, 59, 27]
        t7 = [55, 320, 56, 104, 220, 239, 47, 246, 176, 182, 33]
        t8 = [23, 261, 87, 7, 120, 14, 62, 47, 225, 71, 246, 21, 42, 20, 5,
              12, 120, 11, 3, 14, 71, 11, 14, 11, 16, 90, 1, 16, 52, 95]
        t9 = [97, 51, 11, 4, 141, 18, 142, 68, 77, 80, 1, 16, 106, 206, 82,
              54, 31, 216, 46, 111, 39, 63, 18, 191, 18, 163, 24]
        t10 = [50, 44, 102, 72, 22, 39, 3, 15, 197, 188, 79, 88, 46, 5, 5, 36,
               22, 139, 210, 97, 30, 23, 13, 14]
        t11 = [359, 9, 12, 270, 603, 3, 104, 2, 438]
        t12 = [50, 254, 5, 283, 35, 12]
        t13 = [487, 18, 100, 7, 98, 5, 85, 91, 43, 230, 3, 130]
        t14 = [102, 209, 14, 57, 54, 32, 67, 59, 134, 152, 27, 14, 230, 66,
               61, 34]

        Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4, t5, t6, t7, t8,
                                          t9, t10, t11, t12, t13, t14),
                                         midrank=False)
        assert_almost_equal(Tk, 3.288, 3)
        assert_array_almost_equal([0.5990, 1.3269, 1.8052, 2.2486, 2.8009],
                                  tm[0:5], 4)
        assert_allclose(p, 0.0041, atol=0.00025)

    def test_example2b(self):
        # Example data taken from an earlier technical report of
        # Scholz and Stephens
        t1 = [194, 15, 41, 29, 33, 181]
        t2 = [413, 14, 58, 37, 100, 65, 9, 169, 447, 184, 36, 201, 118]
        t3 = [34, 31, 18, 18, 67, 57, 62, 7, 22, 34]
        t4 = [90, 10, 60, 186, 61, 49, 14, 24, 56, 20, 79, 84, 44, 59, 29,
              118, 25, 156, 310, 76, 26, 44, 23, 62]
        t5 = [130, 208, 70, 101, 208]
        t6 = [74, 57, 48, 29, 502, 12, 70, 21, 29, 386, 59, 27]
        t7 = [55, 320, 56, 104, 220, 239, 47, 246, 176, 182, 33]
        t8 = [23, 261, 87, 7, 120, 14, 62, 47, 225, 71, 246, 21, 42, 20, 5,
              12, 120, 11, 3, 14, 71, 11, 14, 11, 16, 90, 1, 16, 52, 95]
        t9 = [97, 51, 11, 4, 141, 18, 142, 68, 77, 80, 1, 16, 106, 206, 82,
              54, 31, 216, 46, 111, 39, 63, 18, 191, 18, 163, 24]
        t10 = [50, 44, 102, 72, 22, 39, 3, 15, 197, 188, 79, 88, 46, 5, 5, 36,
               22, 139, 210, 97, 30, 23, 13, 14]
        t11 = [359, 9, 12, 270, 603, 3, 104, 2, 438]
        t12 = [50, 254, 5, 283, 35, 12]
        t13 = [487, 18, 100, 7, 98, 5, 85, 91, 43, 230, 3, 130]
        t14 = [102, 209, 14, 57, 54, 32, 67, 59, 134, 152, 27, 14, 230, 66,
               61, 34]

        Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4, t5, t6, t7, t8,
                                          t9, t10, t11, t12, t13, t14),
                                         midrank=True)

        assert_almost_equal(Tk, 3.294, 3)
        assert_array_almost_equal([0.5990, 1.3269, 1.8052, 2.2486, 2.8009],
                                  tm[0:5], 4)
        assert_allclose(p, 0.0041, atol=0.00025)

    def test_R_kSamples(self):
        # test values generates with R package kSamples
        # package version 1.2-6 (2017-06-14)
        # r1 = 1:100
        # continuous case (no ties) --> version  1
        # res <- kSamples::ad.test(r1, r1 + 40.5)
        # res$ad[1, "T.AD"] #  41.105
        # res$ad[1, " asympt. P-value"] #  5.8399e-18
        #
        # discrete case (ties allowed) --> version  2 (here: midrank=True)
        # res$ad[2, "T.AD"] #  41.235
        #
        # res <- kSamples::ad.test(r1, r1 + .5)
        # res$ad[1, "T.AD"] #  -1.2824
        # res$ad[1, " asympt. P-value"] #  1
        # res$ad[2, "T.AD"] #  -1.2944
        #
        # res <- kSamples::ad.test(r1, r1 + 7.5)
        # res$ad[1, "T.AD"] # 1.4923
        # res$ad[1, " asympt. P-value"] # 0.077501
        #
        # res <- kSamples::ad.test(r1, r1 + 6)
        # res$ad[2, "T.AD"] # 0.63892
        # res$ad[2, " asympt. P-value"] # 0.17981
        #
        # res <- kSamples::ad.test(r1, r1 + 11.5)
        # res$ad[1, "T.AD"] # 4.5042
        # res$ad[1, " asympt. P-value"] # 0.00545
        #
        # res <- kSamples::ad.test(r1, r1 + 13.5)
        # res$ad[1, "T.AD"] # 6.2982
        # res$ad[1, " asympt. P-value"] # 0.00118

        x1 = np.linspace(1, 100, 100)
        # test case: different distributions;p-value floored at 0.001
        # test case for issue #5493 / #8536
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='p-value floored')
            s, _, p = stats.anderson_ksamp([x1, x1 + 40.5], midrank=False)
        assert_almost_equal(s, 41.105, 3)
        assert_equal(p, 0.001)

        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='p-value floored')
            s, _, p = stats.anderson_ksamp([x1, x1 + 40.5])
        assert_almost_equal(s, 41.235, 3)
        assert_equal(p, 0.001)

        # test case: similar distributions --> p-value capped at 0.25
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='p-value capped')
            s, _, p = stats.anderson_ksamp([x1, x1 + .5], midrank=False)
        assert_almost_equal(s, -1.2824, 4)
        assert_equal(p, 0.25)

        with suppress_warnings() as sup:
            sup.filter(UserWarning, message='p-value capped')
            s, _, p = stats.anderson_ksamp([x1, x1 + .5])
        assert_almost_equal(s, -1.2944, 4)
        assert_equal(p, 0.25)

        # test case: check interpolated p-value in [0.01, 0.25] (no ties)
        s, _, p = stats.anderson_ksamp([x1, x1 + 7.5], midrank=False)
        assert_almost_equal(s, 1.4923, 4)
        assert_allclose(p, 0.0775, atol=0.005, rtol=0)

        # test case: check interpolated p-value in [0.01, 0.25] (w/ ties)
        s, _, p = stats.anderson_ksamp([x1, x1 + 6])
        assert_almost_equal(s, 0.6389, 4)
        assert_allclose(p, 0.1798, atol=0.005, rtol=0)

        # test extended critical values for p=0.001 and p=0.005
        s, _, p = stats.anderson_ksamp([x1, x1 + 11.5], midrank=False)
        assert_almost_equal(s, 4.5042, 4)
        assert_allclose(p, 0.00545, atol=0.0005, rtol=0)

        s, _, p = stats.anderson_ksamp([x1, x1 + 13.5], midrank=False)
        assert_almost_equal(s, 6.2982, 4)
        assert_allclose(p, 0.00118, atol=0.0001, rtol=0)

    def test_not_enough_samples(self):
        assert_raises(ValueError, stats.anderson_ksamp, np.ones(5))

    def test_no_distinct_observations(self):
        assert_raises(ValueError, stats.anderson_ksamp,
                      (np.ones(5), np.ones(5)))

    def test_empty_sample(self):
        assert_raises(ValueError, stats.anderson_ksamp, (np.ones(5), []))

    def test_result_attributes(self):
        # Pass a mixture of lists and arrays
        t1 = [38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0]
        t2 = np.array([39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8])
        res = stats.anderson_ksamp((t1, t2), midrank=False)

        attributes = ('statistic', 'critical_values', 'significance_level')
        check_named_results(res, attributes)


class TestAnsari:

    def test_small(self):
        x = [1, 2, 3, 3, 4]
        y = [3, 2, 6, 1, 6, 1, 4, 1]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "Ties preclude use of exact statistic.")
            W, pval = stats.ansari(x, y)
        assert_almost_equal(W, 23.5, 11)
        assert_almost_equal(pval, 0.13499256881897437, 11)

    def test_approx(self):
        ramsay = np.array((111, 107, 100, 99, 102, 106, 109, 108, 104, 99,
                           101, 96, 97, 102, 107, 113, 116, 113, 110, 98))
        parekh = np.array((107, 108, 106, 98, 105, 103, 110, 105, 104,
                           100, 96, 108, 103, 104, 114, 114, 113, 108,
                           106, 99))

        with suppress_warnings() as sup:
            sup.filter(UserWarning, "Ties preclude use of exact statistic.")
            W, pval = stats.ansari(ramsay, parekh)

        assert_almost_equal(W, 185.5, 11)
        assert_almost_equal(pval, 0.18145819972867083, 11)

    def test_exact(self):
        W, pval = stats.ansari([1, 2, 3, 4], [15, 5, 20, 8, 10, 12])
        assert_almost_equal(W, 10.0, 11)
        assert_almost_equal(pval, 0.533333333333333333, 7)

    def test_bad_arg(self):
        assert_raises(ValueError, stats.ansari, [], [1])
        assert_raises(ValueError, stats.ansari, [1], [])

    def test_result_attributes(self):
        x = [1, 2, 3, 3, 4]
        y = [3, 2, 6, 1, 6, 1, 4, 1]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "Ties preclude use of exact statistic.")
            res = stats.ansari(x, y)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_bad_alternative(self):
        # invalid value for alternative must raise a ValueError
        x1 = [1, 2, 3, 4]
        x2 = [5, 6, 7, 8]
        match = "'alternative' must be 'two-sided'"
        with assert_raises(ValueError, match=match):
            stats.ansari(x1, x2, alternative='foo')

    def test_alternative_exact(self):
        x1 = [-5, 1, 5, 10, 15, 20, 25] # high scale, loc=10
        x2 = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5] # low scale, loc=10
        # ratio of scales is greater than 1. So, the
        # p-value must be high when `alternative='less'`
        # and low when `alternative='greater'`.
        statistic, pval = stats.ansari(x1, x2)
        pval_l = stats.ansari(x1, x2, alternative='less').pvalue
        pval_g = stats.ansari(x1, x2, alternative='greater').pvalue
        assert pval_l > 0.95
        assert pval_g < 0.05 # level of significance.
        # also check if the p-values sum up to 1 plus the the probability
        # mass under the calculated statistic.
        prob = _abw_state.pmf(statistic, len(x1), len(x2))
        assert_allclose(pval_g + pval_l, 1 + prob, atol=1e-12)
        # also check if one of the one-sided p-value equals half the
        # two-sided p-value and the other one-sided p-value is its
        # compliment.
        assert_allclose(pval_g, pval/2, atol=1e-12)
        assert_allclose(pval_l, 1+prob-pval/2, atol=1e-12)
        # sanity check. The result should flip if
        # we exchange x and y.
        pval_l_reverse = stats.ansari(x2, x1, alternative='less').pvalue
        pval_g_reverse = stats.ansari(x2, x1, alternative='greater').pvalue
        assert pval_l_reverse < 0.05
        assert pval_g_reverse > 0.95

    @pytest.mark.parametrize(
        'x, y, alternative, expected',
        # the tests are designed in such a way that the
        # if else statement in ansari test for exact
        # mode is covered.
        [([1, 2, 3, 4], [5, 6, 7, 8], 'less', 0.6285714285714),
         ([1, 2, 3, 4], [5, 6, 7, 8], 'greater', 0.6285714285714),
         ([1, 2, 3], [4, 5, 6, 7, 8], 'less', 0.8928571428571),
         ([1, 2, 3], [4, 5, 6, 7, 8], 'greater', 0.2857142857143),
         ([1, 2, 3, 4, 5], [6, 7, 8], 'less', 0.2857142857143),
         ([1, 2, 3, 4, 5], [6, 7, 8], 'greater', 0.8928571428571)]
    )
    def test_alternative_exact_with_R(self, x, y, alternative, expected):
        # testing with R on arbitrary data
        # Sample R code used for the third test case above:
        # ```R
        # > options(digits=16)
        # > x <- c(1,2,3)
        # > y <- c(4,5,6,7,8)
        # > ansari.test(x, y, alternative='less', exact=TRUE)
        #
        #     Ansari-Bradley test
        #
        # data:  x and y
        # AB = 6, p-value = 0.8928571428571
        # alternative hypothesis: true ratio of scales is less than 1
        #
        # ```
        pval = stats.ansari(x, y, alternative=alternative).pvalue
        assert_allclose(pval, expected, atol=1e-12)

    def test_alternative_approx(self):
        # intuitive tests for approximation
        x1 = stats.norm.rvs(0, 5, size=100, random_state=123)
        x2 = stats.norm.rvs(0, 2, size=100, random_state=123)
        # for m > 55 or n > 55, the test should automatically
        # switch to approximation.
        pval_l = stats.ansari(x1, x2, alternative='less').pvalue
        pval_g = stats.ansari(x1, x2, alternative='greater').pvalue
        assert_allclose(pval_l, 1.0, atol=1e-12)
        assert_allclose(pval_g, 0.0, atol=1e-12)
        # also check if one of the one-sided p-value equals half the
        # two-sided p-value and the other one-sided p-value is its
        # compliment.
        x1 = stats.norm.rvs(0, 2, size=60, random_state=123)
        x2 = stats.norm.rvs(0, 1.5, size=60, random_state=123)
        pval = stats.ansari(x1, x2).pvalue
        pval_l = stats.ansari(x1, x2, alternative='less').pvalue
        pval_g = stats.ansari(x1, x2, alternative='greater').pvalue
        assert_allclose(pval_g, pval/2, atol=1e-12)
        assert_allclose(pval_l, 1-pval/2, atol=1e-12)


class TestBartlett:

    def test_data(self):
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda357.htm
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        T, pval = stats.bartlett(*args)
        assert_almost_equal(T, 20.78587342806484, 7)
        assert_almost_equal(pval, 0.0136358632781, 7)

    def test_bad_arg(self):
        # Too few args raises ValueError.
        assert_raises(ValueError, stats.bartlett, [1])

    def test_result_attributes(self):
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        res = stats.bartlett(*args)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_empty_arg(self):
        args = (g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, [])
        assert_equal((np.nan, np.nan), stats.bartlett(*args))

    # temporary fix for issue #9252: only accept 1d input
    def test_1d_input(self):
        x = np.array([[1, 2], [3, 4]])
        assert_raises(ValueError, stats.bartlett, g1, x)


class TestLevene:

    def test_data(self):
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        W, pval = stats.levene(*args)
        assert_almost_equal(W, 1.7059176930008939, 7)
        assert_almost_equal(pval, 0.0990829755522, 7)

    def test_trimmed1(self):
        # Test that center='trimmed' gives the same result as center='mean'
        # when proportiontocut=0.
        W1, pval1 = stats.levene(g1, g2, g3, center='mean')
        W2, pval2 = stats.levene(g1, g2, g3, center='trimmed',
                                 proportiontocut=0.0)
        assert_almost_equal(W1, W2)
        assert_almost_equal(pval1, pval2)

    def test_trimmed2(self):
        x = [1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
        y = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 200.0]
        np.random.seed(1234)
        x2 = np.random.permutation(x)

        # Use center='trimmed'
        W0, pval0 = stats.levene(x, y, center='trimmed',
                                 proportiontocut=0.125)
        W1, pval1 = stats.levene(x2, y, center='trimmed',
                                 proportiontocut=0.125)
        # Trim the data here, and use center='mean'
        W2, pval2 = stats.levene(x[1:-1], y[1:-1], center='mean')
        # Result should be the same.
        assert_almost_equal(W0, W2)
        assert_almost_equal(W1, W2)
        assert_almost_equal(pval1, pval2)

    def test_equal_mean_median(self):
        x = np.linspace(-1, 1, 21)
        np.random.seed(1234)
        x2 = np.random.permutation(x)
        y = x**3
        W1, pval1 = stats.levene(x, y, center='mean')
        W2, pval2 = stats.levene(x2, y, center='median')
        assert_almost_equal(W1, W2)
        assert_almost_equal(pval1, pval2)

    def test_bad_keyword(self):
        x = np.linspace(-1, 1, 21)
        assert_raises(TypeError, stats.levene, x, x, portiontocut=0.1)

    def test_bad_center_value(self):
        x = np.linspace(-1, 1, 21)
        assert_raises(ValueError, stats.levene, x, x, center='trim')

    def test_too_few_args(self):
        assert_raises(ValueError, stats.levene, [1])

    def test_result_attributes(self):
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        res = stats.levene(*args)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    # temporary fix for issue #9252: only accept 1d input
    def test_1d_input(self):
        x = np.array([[1, 2], [3, 4]])
        assert_raises(ValueError, stats.levene, g1, x)


class TestBinomP:
    """Tests for stats.binom_test."""

    binom_test_func = staticmethod(stats.binom_test)

    def test_data(self):
        pval = self.binom_test_func(100, 250)
        assert_almost_equal(pval, 0.0018833009350757682, 11)
        pval = self.binom_test_func(201, 405)
        assert_almost_equal(pval, 0.92085205962670713, 11)
        pval = self.binom_test_func([682, 243], p=3/4)
        assert_almost_equal(pval, 0.38249155957481695, 11)

    def test_bad_len_x(self):
        # Length of x must be 1 or 2.
        assert_raises(ValueError, self.binom_test_func, [1, 2, 3])

    def test_bad_n(self):
        # len(x) is 1, but n is invalid.
        # Missing n
        assert_raises(ValueError, self.binom_test_func, [100])
        # n less than x[0]
        assert_raises(ValueError, self.binom_test_func, [100], n=50)

    def test_bad_p(self):
        assert_raises(ValueError,
                      self.binom_test_func, [50, 50], p=2.0)

    def test_alternatives(self):
        res = self.binom_test_func(51, 235, p=1/6, alternative='less')
        assert_almost_equal(res, 0.982022657605858)

        res = self.binom_test_func(51, 235, p=1/6, alternative='greater')
        assert_almost_equal(res, 0.02654424571169085)

        res = self.binom_test_func(51, 235, p=1/6, alternative='two-sided')
        assert_almost_equal(res, 0.0437479701823997)


class TestBinomTestP(TestBinomP):
    """
    Tests for stats.binomtest as a replacement for stats.binom_test.
    """
    @staticmethod
    def binom_test_func(x, n=None, p=0.5, alternative='two-sided'):
        # This processing of x and n is copied from from binom_test.
        x = np.atleast_1d(x).astype(np.int_)
        if len(x) == 2:
            n = x[1] + x[0]
            x = x[0]
        elif len(x) == 1:
            x = x[0]
            if n is None or n < x:
                raise ValueError("n must be >= x")
            n = np.int_(n)
        else:
            raise ValueError("Incorrect length for x.")

        result = stats.binomtest(x, n, p=p, alternative=alternative)
        return result.pvalue


class TestBinomTest:
    """Tests for stats.binomtest."""

    # Expected results here are from R binom.test, e.g.
    # options(digits=16)
    # binom.test(484, 967, p=0.48)
    #
    def test_two_sided_pvalues1(self):
        # `tol` could be stricter on most architectures, but the value
        # here is limited by accuracy of `binom.cdf` for large inputs on
        # Linux_Python_37_32bit_full and aarch64
        rtol = 1e-10  # aarch64 observed rtol: 1.5e-11
        res = stats.binomtest(10079999, 21000000, 0.48)
        assert_allclose(res.pvalue, 1.0, rtol=rtol)
        res = stats.binomtest(10079990, 21000000, 0.48)
        assert_allclose(res.pvalue, 0.9966892187965, rtol=rtol)
        res = stats.binomtest(10080009, 21000000, 0.48)
        assert_allclose(res.pvalue, 0.9970377203856, rtol=rtol)
        res = stats.binomtest(10080017, 21000000, 0.48)
        assert_allclose(res.pvalue, 0.9940754817328, rtol=1e-9)

    def test_two_sided_pvalues2(self):
        rtol = 1e-10  # no aarch64 failure with 1e-15, preemptive bump
        res = stats.binomtest(9, n=21, p=0.48)
        assert_allclose(res.pvalue, 0.6689672431939, rtol=rtol)
        res = stats.binomtest(4, 21, 0.48)
        assert_allclose(res.pvalue, 0.008139563452106, rtol=rtol)
        res = stats.binomtest(11, 21, 0.48)
        assert_allclose(res.pvalue, 0.8278629664608, rtol=rtol)
        res = stats.binomtest(7, 21, 0.48)
        assert_allclose(res.pvalue, 0.1966772901718, rtol=rtol)
        res = stats.binomtest(3, 10, .5)
        assert_allclose(res.pvalue, 0.34375, rtol=rtol)
        res = stats.binomtest(2, 2, .4)
        assert_allclose(res.pvalue, 0.16, rtol=rtol)
        res = stats.binomtest(2, 4, .3)
        assert_allclose(res.pvalue, 0.5884, rtol=rtol)

    def test_edge_cases(self):
        rtol = 1e-10  # aarch64 observed rtol: 1.33e-15
        res = stats.binomtest(484, 967, 0.5)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(3, 47, 3/47)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(13, 46, 13/46)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(15, 44, 15/44)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(7, 13, 0.5)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(6, 11, 0.5)
        assert_allclose(res.pvalue, 1, rtol=rtol)

    def test_binary_srch_for_binom_tst(self):
        # Test that old behavior of binomtest is maintained
        # by the new binary search method in cases where d
        # exactly equals the input on one side.
        n = 10
        p = 0.5
        k = 3
        # First test for the case where k > mode of PMF
        i = np.arange(np.ceil(p * n), n+1)
        d = stats.binom.pmf(k, n, p)
        # Old way of calculating y, probably consistent with R.
        y1 = np.sum(stats.binom.pmf(i, n, p) <= d, axis=0)
        # New way with binary search.
        ix = _binary_search_for_binom_tst(lambda x1:
                                          -stats.binom.pmf(x1, n, p),
                                          -d, np.ceil(p * n), n)
        y2 = n - ix + int(d == stats.binom.pmf(ix, n, p))
        assert_allclose(y1, y2, rtol=1e-9)
        # Now test for the other side.
        k = 7
        i = np.arange(np.floor(p * n) + 1)
        d = stats.binom.pmf(k, n, p)
        # Old way of calculating y.
        y1 = np.sum(stats.binom.pmf(i, n, p) <= d, axis=0)
        # New way with binary search.
        ix = _binary_search_for_binom_tst(lambda x1:
                                          stats.binom.pmf(x1, n, p),
                                          d, 0, np.floor(p * n))
        y2 = ix + 1
        assert_allclose(y1, y2, rtol=1e-9)

    # Expected results here are from R 3.6.2 binom.test
    @pytest.mark.parametrize('alternative, pval, ci_low, ci_high',
                             [('less', 0.148831050443,
                               0.0, 0.2772002496709138),
                              ('greater', 0.9004695898947,
                               0.1366613252458672, 1.0),
                              ('two-sided', 0.2983720970096,
                               0.1266555521019559, 0.2918426890886281)])
    def test_confidence_intervals1(self, alternative, pval, ci_low, ci_high):
        res = stats.binomtest(20, n=100, p=0.25, alternative=alternative)
        assert_allclose(res.pvalue, pval, rtol=1e-12)
        assert_equal(res.proportion_estimate, 0.2)
        ci = res.proportion_ci(confidence_level=0.95)
        assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-12)

    # Expected results here are from R 3.6.2 binom.test.
    @pytest.mark.parametrize('alternative, pval, ci_low, ci_high',
                             [('less',
                               0.005656361, 0.0, 0.1872093),
                              ('greater',
                               0.9987146, 0.008860761, 1.0),
                              ('two-sided',
                               0.01191714, 0.006872485, 0.202706269)])
    def test_confidence_intervals2(self, alternative, pval, ci_low, ci_high):
        res = stats.binomtest(3, n=50, p=0.2, alternative=alternative)
        assert_allclose(res.pvalue, pval, rtol=1e-6)
        assert_equal(res.proportion_estimate, 0.06)
        ci = res.proportion_ci(confidence_level=0.99)
        assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-6)

    # Expected results here are from R 3.6.2 binom.test.
    @pytest.mark.parametrize('alternative, pval, ci_high',
                             [('less', 0.05631351, 0.2588656),
                              ('greater', 1.0, 1.0),
                              ('two-sided', 0.07604122, 0.3084971)])
    def test_confidence_interval_exact_k0(self, alternative, pval, ci_high):
        # Test with k=0, n = 10.
        res = stats.binomtest(0, 10, p=0.25, alternative=alternative)
        assert_allclose(res.pvalue, pval, rtol=1e-6)
        ci = res.proportion_ci(confidence_level=0.95)
        assert_equal(ci.low, 0.0)
        assert_allclose(ci.high, ci_high, rtol=1e-6)

    # Expected results here are from R 3.6.2 binom.test.
    @pytest.mark.parametrize('alternative, pval, ci_low',
                             [('less', 1.0, 0.0),
                              ('greater', 9.536743e-07, 0.7411344),
                              ('two-sided', 9.536743e-07, 0.6915029)])
    def test_confidence_interval_exact_k_is_n(self, alternative, pval, ci_low):
        # Test with k = n = 10.
        res = stats.binomtest(10, 10, p=0.25, alternative=alternative)
        assert_allclose(res.pvalue, pval, rtol=1e-6)
        ci = res.proportion_ci(confidence_level=0.95)
        assert_equal(ci.high, 1.0)
        assert_allclose(ci.low, ci_low, rtol=1e-6)

    # Expected results are from the prop.test function in R 3.6.2.
    @pytest.mark.parametrize(
        'k, alternative, corr, conf, ci_low, ci_high',
        [[3, 'two-sided', True, 0.95, 0.08094782, 0.64632928],
         [3, 'two-sided', True, 0.99, 0.0586329, 0.7169416],
         [3, 'two-sided', False, 0.95, 0.1077913, 0.6032219],
         [3, 'two-sided', False, 0.99, 0.07956632, 0.6799753],
         [3, 'less', True, 0.95, 0.0, 0.6043476],
         [3, 'less', True, 0.99, 0.0, 0.6901811],
         [3, 'less', False, 0.95, 0.0, 0.5583002],
         [3, 'less', False, 0.99, 0.0, 0.6507187],
         [3, 'greater', True, 0.95, 0.09644904, 1.0],
         [3, 'greater', True, 0.99, 0.06659141, 1.0],
         [3, 'greater', False, 0.95, 0.1268766, 1.0],
         [3, 'greater', False, 0.99, 0.08974147, 1.0],

         [0, 'two-sided', True, 0.95, 0.0, 0.3445372],
         [0, 'two-sided', False, 0.95, 0.0, 0.2775328],
         [0, 'less', True, 0.95, 0.0, 0.2847374],
         [0, 'less', False, 0.95, 0.0, 0.212942],
         [0, 'greater', True, 0.95, 0.0, 1.0],
         [0, 'greater', False, 0.95, 0.0, 1.0],

         [10, 'two-sided', True, 0.95, 0.6554628, 1.0],
         [10, 'two-sided', False, 0.95, 0.7224672, 1.0],
         [10, 'less', True, 0.95, 0.0, 1.0],
         [10, 'less', False, 0.95, 0.0, 1.0],
         [10, 'greater', True, 0.95, 0.7152626, 1.0],
         [10, 'greater', False, 0.95, 0.787058, 1.0]]
    )
    def test_ci_wilson_method(self, k, alternative, corr, conf,
                              ci_low, ci_high):
        res = stats.binomtest(k, n=10, p=0.1, alternative=alternative)
        if corr:
            method = 'wilsoncc'
        else:
            method = 'wilson'
        ci = res.proportion_ci(confidence_level=conf, method=method)
        assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-6)

    def test_estimate_equals_hypothesized_prop(self):
        # Test the special case where the estimated proportion equals
        # the hypothesized proportion.  When alternative is 'two-sided',
        # the p-value is 1.
        res = stats.binomtest(4, 16, 0.25)
        assert_equal(res.proportion_estimate, 0.25)
        assert_equal(res.pvalue, 1.0)

    @pytest.mark.parametrize('k, n', [(0, 0), (-1, 2)])
    def test_invalid_k_n(self, k, n):
        with pytest.raises(ValueError,
                           match="must be an integer not less than"):
            stats.binomtest(k, n)

    def test_invalid_k_too_big(self):
        with pytest.raises(ValueError,
                           match="k must not be greater than n"):
            stats.binomtest(11, 10, 0.25)

    def test_invalid_confidence_level(self):
        res = stats.binomtest(3, n=10, p=0.1)
        with pytest.raises(ValueError, match="must be in the interval"):
            res.proportion_ci(confidence_level=-1)

    def test_invalid_ci_method(self):
        res = stats.binomtest(3, n=10, p=0.1)
        with pytest.raises(ValueError, match="method must be"):
            res.proportion_ci(method="plate of shrimp")


class TestFligner:

    def test_data(self):
        # numbers from R: fligner.test in package stats
        x1 = np.arange(5)
        assert_array_almost_equal(stats.fligner(x1, x1**2),
                                  (3.2282229927203536, 0.072379187848207877),
                                  11)

    def test_trimmed1(self):
        # Perturb input to break ties in the transformed data
        # See https://github.com/scipy/scipy/pull/8042 for more details
        rs = np.random.RandomState(123)
        _perturb = lambda g: (np.asarray(g) + 1e-10*rs.randn(len(g))).tolist()
        g1_ = _perturb(g1)
        g2_ = _perturb(g2)
        g3_ = _perturb(g3)
        # Test that center='trimmed' gives the same result as center='mean'
        # when proportiontocut=0.
        Xsq1, pval1 = stats.fligner(g1_, g2_, g3_, center='mean')
        Xsq2, pval2 = stats.fligner(g1_, g2_, g3_, center='trimmed',
                                    proportiontocut=0.0)
        assert_almost_equal(Xsq1, Xsq2)
        assert_almost_equal(pval1, pval2)

    def test_trimmed2(self):
        x = [1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
        y = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 200.0]
        # Use center='trimmed'
        Xsq1, pval1 = stats.fligner(x, y, center='trimmed',
                                    proportiontocut=0.125)
        # Trim the data here, and use center='mean'
        Xsq2, pval2 = stats.fligner(x[1:-1], y[1:-1], center='mean')
        # Result should be the same.
        assert_almost_equal(Xsq1, Xsq2)
        assert_almost_equal(pval1, pval2)

    # The following test looks reasonable at first, but fligner() uses the
    # function stats.rankdata(), and in one of the cases in this test,
    # there are ties, while in the other (because of normal rounding
    # errors) there are not.  This difference leads to differences in the
    # third significant digit of W.
    #
    #def test_equal_mean_median(self):
    #    x = np.linspace(-1,1,21)
    #    y = x**3
    #    W1, pval1 = stats.fligner(x, y, center='mean')
    #    W2, pval2 = stats.fligner(x, y, center='median')
    #    assert_almost_equal(W1, W2)
    #    assert_almost_equal(pval1, pval2)

    def test_bad_keyword(self):
        x = np.linspace(-1, 1, 21)
        assert_raises(TypeError, stats.fligner, x, x, portiontocut=0.1)

    def test_bad_center_value(self):
        x = np.linspace(-1, 1, 21)
        assert_raises(ValueError, stats.fligner, x, x, center='trim')

    def test_bad_num_args(self):
        # Too few args raises ValueError.
        assert_raises(ValueError, stats.fligner, [1])

    def test_empty_arg(self):
        x = np.arange(5)
        assert_equal((np.nan, np.nan), stats.fligner(x, x**2, []))


class TestMood:
    def test_mood(self):
        # numbers from R: mood.test in package stats
        x1 = np.arange(5)
        assert_array_almost_equal(stats.mood(x1, x1**2),
                                  (-1.3830857299399906, 0.16663858066771478),
                                  11)

    def test_mood_order_of_args(self):
        # z should change sign when the order of arguments changes, pvalue
        # should not change
        np.random.seed(1234)
        x1 = np.random.randn(10, 1)
        x2 = np.random.randn(15, 1)
        z1, p1 = stats.mood(x1, x2)
        z2, p2 = stats.mood(x2, x1)
        assert_array_almost_equal([z1, p1], [-z2, p2])

    def test_mood_with_axis_none(self):
        # Test with axis = None, compare with results from R
        x1 = [-0.626453810742332, 0.183643324222082, -0.835628612410047,
               1.59528080213779, 0.329507771815361, -0.820468384118015,
               0.487429052428485, 0.738324705129217, 0.575781351653492,
              -0.305388387156356, 1.51178116845085, 0.389843236411431,
              -0.621240580541804, -2.2146998871775, 1.12493091814311,
              -0.0449336090152309, -0.0161902630989461, 0.943836210685299,
               0.821221195098089, 0.593901321217509]

        x2 = [-0.896914546624981, 0.184849184646742, 1.58784533120882,
              -1.13037567424629, -0.0802517565509893, 0.132420284381094,
               0.707954729271733, -0.23969802417184, 1.98447393665293,
              -0.138787012119665, 0.417650750792556, 0.981752777463662,
              -0.392695355503813, -1.03966897694891, 1.78222896030858,
              -2.31106908460517, 0.878604580921265, 0.035806718015226,
               1.01282869212708, 0.432265154539617, 2.09081920524915,
              -1.19992581964387, 1.58963820029007, 1.95465164222325,
               0.00493777682814261, -2.45170638784613, 0.477237302613617,
              -0.596558168631403, 0.792203270299649, 0.289636710177348]

        x1 = np.array(x1)
        x2 = np.array(x2)
        x1.shape = (10, 2)
        x2.shape = (15, 2)
        assert_array_almost_equal(stats.mood(x1, x2, axis=None),
                                  [-1.31716607555, 0.18778296257])

    def test_mood_2d(self):
        # Test if the results of mood test in 2-D case are consistent with the
        # R result for the same inputs.  Numbers from R mood.test().
        ny = 5
        np.random.seed(1234)
        x1 = np.random.randn(10, ny)
        x2 = np.random.randn(15, ny)
        z_vectest, pval_vectest = stats.mood(x1, x2)

        for j in range(ny):
            assert_array_almost_equal([z_vectest[j], pval_vectest[j]],
                                      stats.mood(x1[:, j], x2[:, j]))

        # inverse order of dimensions
        x1 = x1.transpose()
        x2 = x2.transpose()
        z_vectest, pval_vectest = stats.mood(x1, x2, axis=1)

        for i in range(ny):
            # check axis handling is self consistent
            assert_array_almost_equal([z_vectest[i], pval_vectest[i]],
                                      stats.mood(x1[i, :], x2[i, :]))

    def test_mood_3d(self):
        shape = (10, 5, 6)
        np.random.seed(1234)
        x1 = np.random.randn(*shape)
        x2 = np.random.randn(*shape)

        for axis in range(3):
            z_vectest, pval_vectest = stats.mood(x1, x2, axis=axis)
            # Tests that result for 3-D arrays is equal to that for the
            # same calculation on a set of 1-D arrays taken from the
            # 3-D array
            axes_idx = ([1, 2], [0, 2], [0, 1])  # the two axes != axis
            for i in range(shape[axes_idx[axis][0]]):
                for j in range(shape[axes_idx[axis][1]]):
                    if axis == 0:
                        slice1 = x1[:, i, j]
                        slice2 = x2[:, i, j]
                    elif axis == 1:
                        slice1 = x1[i, :, j]
                        slice2 = x2[i, :, j]
                    else:
                        slice1 = x1[i, j, :]
                        slice2 = x2[i, j, :]

                    assert_array_almost_equal([z_vectest[i, j],
                                               pval_vectest[i, j]],
                                              stats.mood(slice1, slice2))

    def test_mood_bad_arg(self):
        # Raise ValueError when the sum of the lengths of the args is
        # less than 3
        assert_raises(ValueError, stats.mood, [1], [])

    def test_mood_alternative(self):

        np.random.seed(0)
        x = stats.norm.rvs(scale=0.75, size=100)
        y = stats.norm.rvs(scale=1.25, size=100)

        stat1, p1 = stats.mood(x, y, alternative='two-sided')
        stat2, p2 = stats.mood(x, y, alternative='less')
        stat3, p3 = stats.mood(x, y, alternative='greater')

        assert stat1 == stat2 == stat3
        assert_allclose(p1, 0, atol=1e-7)
        assert_allclose(p2, p1/2)
        assert_allclose(p3, 1 - p1/2)

        with pytest.raises(ValueError, match="alternative must be..."):
            stats.mood(x, y, alternative='ekki-ekki')

    @pytest.mark.xfail(reason="SciPy needs tie correction like R (gh-13730)")
    @pytest.mark.parametrize("alternative, expected",
                             [('two-sided', (1.037127561496, 0.299676411857)),
                              ('less', (1.0371275614961, 0.8501617940715)),
                              ('greater', (1.037127561496, 0.1498382059285))])
    def test_mood_alternative_against_R(self, alternative, expected):
        ## Test againts R mood.test: https://rdrr.io/r/stats/mood.test.html
        # options(digits=16)
        # x <- c(111, 107, 100, 99, 102, 106, 109, 108, 104, 99,
        #             101, 96, 97, 102, 107, 113, 116, 113, 110, 98)
        # y <- c(107, 108, 106, 98, 105, 103, 110, 105, 104,
        #             100, 96, 108, 103, 104, 114, 114, 113, 108, 106, 99)
        # mood.test(x, y, alternative='less')
        x = [111, 107, 100, 99, 102, 106, 109, 108, 104, 99,
             101, 96, 97, 102, 107, 113, 116, 113, 110, 98]
        y = [107, 108, 106, 98, 105, 103, 110, 105, 104, 100,
             96, 108, 103, 104, 114, 114, 113, 108, 106, 99]

        res = stats.mood(x, y, alternative=alternative)
        assert_allclose(res, expected)


class TestProbplot:

    def test_basic(self):
        x = stats.norm.rvs(size=20, random_state=12345)
        osm, osr = stats.probplot(x, fit=False)
        osm_expected = [-1.8241636, -1.38768012, -1.11829229, -0.91222575,
                        -0.73908135, -0.5857176, -0.44506467, -0.31273668,
                        -0.18568928, -0.06158146, 0.06158146, 0.18568928,
                        0.31273668, 0.44506467, 0.5857176, 0.73908135,
                        0.91222575, 1.11829229, 1.38768012, 1.8241636]
        assert_allclose(osr, np.sort(x))
        assert_allclose(osm, osm_expected)

        res, res_fit = stats.probplot(x, fit=True)
        res_fit_expected = [1.05361841, 0.31297795, 0.98741609]
        assert_allclose(res_fit, res_fit_expected)

    def test_sparams_keyword(self):
        x = stats.norm.rvs(size=100, random_state=123456)
        # Check that None, () and 0 (loc=0, for normal distribution) all work
        # and give the same results
        osm1, osr1 = stats.probplot(x, sparams=None, fit=False)
        osm2, osr2 = stats.probplot(x, sparams=0, fit=False)
        osm3, osr3 = stats.probplot(x, sparams=(), fit=False)
        assert_allclose(osm1, osm2)
        assert_allclose(osm1, osm3)
        assert_allclose(osr1, osr2)
        assert_allclose(osr1, osr3)
        # Check giving (loc, scale) params for normal distribution
        osm, osr = stats.probplot(x, sparams=(), fit=False)

    def test_dist_keyword(self):
        x = stats.norm.rvs(size=20, random_state=12345)
        osm1, osr1 = stats.probplot(x, fit=False, dist='t', sparams=(3,))
        osm2, osr2 = stats.probplot(x, fit=False, dist=stats.t, sparams=(3,))
        assert_allclose(osm1, osm2)
        assert_allclose(osr1, osr2)

        assert_raises(ValueError, stats.probplot, x, dist='wrong-dist-name')
        assert_raises(AttributeError, stats.probplot, x, dist=[])

        class custom_dist:
            """Some class that looks just enough like a distribution."""
            def ppf(self, q):
                return stats.norm.ppf(q, loc=2)

        osm1, osr1 = stats.probplot(x, sparams=(2,), fit=False)
        osm2, osr2 = stats.probplot(x, dist=custom_dist(), fit=False)
        assert_allclose(osm1, osm2)
        assert_allclose(osr1, osr2)

    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    def test_plot_kwarg(self):
        fig = plt.figure()
        fig.add_subplot(111)
        x = stats.t.rvs(3, size=100, random_state=7654321)
        res1, fitres1 = stats.probplot(x, plot=plt)
        plt.close()
        res2, fitres2 = stats.probplot(x, plot=None)
        res3 = stats.probplot(x, fit=False, plot=plt)
        plt.close()
        res4 = stats.probplot(x, fit=False, plot=None)
        # Check that results are consistent between combinations of `fit` and
        # `plot` keywords.
        assert_(len(res1) == len(res2) == len(res3) == len(res4) == 2)
        assert_allclose(res1, res2)
        assert_allclose(res1, res3)
        assert_allclose(res1, res4)
        assert_allclose(fitres1, fitres2)

        # Check that a Matplotlib Axes object is accepted
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.probplot(x, fit=False, plot=ax)
        plt.close()

    def test_probplot_bad_args(self):
        # Raise ValueError when given an invalid distribution.
        assert_raises(ValueError, stats.probplot, [1], dist="plate_of_shrimp")

    def test_empty(self):
        assert_equal(stats.probplot([], fit=False),
                     (np.array([]), np.array([])))
        assert_equal(stats.probplot([], fit=True),
                     ((np.array([]), np.array([])),
                      (np.nan, np.nan, 0.0)))

    def test_array_of_size_one(self):
        with np.errstate(invalid='ignore'):
            assert_equal(stats.probplot([1], fit=True),
                         ((np.array([0.]), np.array([1])),
                          (np.nan, np.nan, 0.0)))


class TestWilcoxon:
    def test_wilcoxon_bad_arg(self):
        # Raise ValueError when two args of different lengths are given or
        # zero_method is unknown.
        assert_raises(ValueError, stats.wilcoxon, [1], [1, 2])
        assert_raises(ValueError, stats.wilcoxon, [1, 2], [1, 2], "dummy")
        assert_raises(ValueError, stats.wilcoxon, [1, 2], [1, 2],
                      alternative="dummy")
        assert_raises(ValueError, stats.wilcoxon, [1]*10, mode="xyz")

    def test_zero_diff(self):
        x = np.arange(20)
        # pratt and wilcox do not work if x - y == 0
        assert_raises(ValueError, stats.wilcoxon, x, x, "wilcox",
                      mode="approx")
        assert_raises(ValueError, stats.wilcoxon, x, x, "pratt",
                      mode="approx")
        # ranksum is n*(n+1)/2, split in half if zero_method == "zsplit"
        assert_equal(stats.wilcoxon(x, x, "zsplit", mode="approx"),
                     (20*21/4, 1.0))

    def test_pratt(self):
        # regression test for gh-6805: p-value matches value from R package
        # coin (wilcoxsign_test) reported in the issue
        x = [1, 2, 3, 4]
        y = [1, 2, 3, 5]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Sample size too small")
            res = stats.wilcoxon(x, y, zero_method="pratt", mode="approx")
        assert_allclose(res, (0.0, 0.31731050786291415))

    def test_wilcoxon_arg_type(self):
        # Should be able to accept list as arguments.
        # Address issue 6070.
        arr = [1, 2, 3, 0, -1, 3, 1, 2, 1, 1, 2]

        _ = stats.wilcoxon(arr, zero_method="pratt", mode="approx")
        _ = stats.wilcoxon(arr, zero_method="zsplit", mode="approx")
        _ = stats.wilcoxon(arr, zero_method="wilcox", mode="approx")

    def test_accuracy_wilcoxon(self):
        freq = [1, 4, 16, 15, 8, 4, 5, 1, 2]
        nums = range(-4, 5)
        x = np.concatenate([[u] * v for u, v in zip(nums, freq)])
        y = np.zeros(x.size)

        T, p = stats.wilcoxon(x, y, "pratt", mode="approx")
        assert_allclose(T, 423)
        assert_allclose(p, 0.0031724568006762576)

        T, p = stats.wilcoxon(x, y, "zsplit", mode="approx")
        assert_allclose(T, 441)
        assert_allclose(p, 0.0032145343172473055)

        T, p = stats.wilcoxon(x, y, "wilcox", mode="approx")
        assert_allclose(T, 327)
        assert_allclose(p, 0.00641346115861)

        # Test the 'correction' option, using values computed in R with:
        # > wilcox.test(x, y, paired=TRUE, exact=FALSE, correct={FALSE,TRUE})
        x = np.array([120, 114, 181, 188, 180, 146, 121, 191, 132, 113, 127, 112])
        y = np.array([133, 143, 119, 189, 112, 199, 198, 113, 115, 121, 142, 187])
        T, p = stats.wilcoxon(x, y, correction=False, mode="approx")
        assert_equal(T, 34)
        assert_allclose(p, 0.6948866, rtol=1e-6)
        T, p = stats.wilcoxon(x, y, correction=True, mode="approx")
        assert_equal(T, 34)
        assert_allclose(p, 0.7240817, rtol=1e-6)

    def test_wilcoxon_result_attributes(self):
        x = np.array([120, 114, 181, 188, 180, 146, 121, 191, 132, 113, 127, 112])
        y = np.array([133, 143, 119, 189, 112, 199, 198, 113, 115, 121, 142, 187])
        res = stats.wilcoxon(x, y, correction=False, mode="approx")
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_wilcoxon_tie(self):
        # Regression test for gh-2391.
        # Corresponding R code is:
        #   > result = wilcox.test(rep(0.1, 10), exact=FALSE, correct=FALSE)
        #   > result$p.value
        #   [1] 0.001565402
        #   > result = wilcox.test(rep(0.1, 10), exact=FALSE, correct=TRUE)
        #   > result$p.value
        #   [1] 0.001904195
        stat, p = stats.wilcoxon([0.1] * 10, mode="approx")
        expected_p = 0.001565402
        assert_equal(stat, 0)
        assert_allclose(p, expected_p, rtol=1e-6)

        stat, p = stats.wilcoxon([0.1] * 10, correction=True, mode="approx")
        expected_p = 0.001904195
        assert_equal(stat, 0)
        assert_allclose(p, expected_p, rtol=1e-6)

    def test_onesided(self):
        # tested against "R version 3.4.1 (2017-06-30)"
        # x <- c(125, 115, 130, 140, 140, 115, 140, 125, 140, 135)
        # y <- c(110, 122, 125, 120, 140, 124, 123, 137, 135, 145)
        # cfg <- list(x = x, y = y, paired = TRUE, exact = FALSE)
        # do.call(wilcox.test, c(cfg, list(alternative = "less", correct = FALSE)))
        # do.call(wilcox.test, c(cfg, list(alternative = "less", correct = TRUE)))
        # do.call(wilcox.test, c(cfg, list(alternative = "greater", correct = FALSE)))
        # do.call(wilcox.test, c(cfg, list(alternative = "greater", correct = TRUE)))
        x = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
        y = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]

        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Sample size too small")
            w, p = stats.wilcoxon(x, y, alternative="less", mode="approx")
        assert_equal(w, 27)
        assert_almost_equal(p, 0.7031847, decimal=6)

        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Sample size too small")
            w, p = stats.wilcoxon(x, y, alternative="less", correction=True,
                                  mode="approx")
        assert_equal(w, 27)
        assert_almost_equal(p, 0.7233656, decimal=6)

        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Sample size too small")
            w, p = stats.wilcoxon(x, y, alternative="greater", mode="approx")
        assert_equal(w, 27)
        assert_almost_equal(p, 0.2968153, decimal=6)

        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Sample size too small")
            w, p = stats.wilcoxon(x, y, alternative="greater", correction=True,
                                  mode="approx")
        assert_equal(w, 27)
        assert_almost_equal(p, 0.3176447, decimal=6)

    def test_exact_basic(self):
        for n in range(1, 26):
            cnt = _get_wilcoxon_distr(n)
            assert_equal(n*(n+1)/2 + 1, len(cnt))
            assert_equal(sum(cnt), 2**n)

    def test_exact_pval(self):
        # expected values computed with "R version 3.4.1 (2017-06-30)"
        x = np.array([1.81, 0.82, 1.56, -0.48, 0.81, 1.28, -1.04, 0.23,
                      -0.75, 0.14])
        y = np.array([0.71, 0.65, -0.2, 0.85, -1.1, -0.45, -0.84, -0.24,
                      -0.68, -0.76])
        _, p = stats.wilcoxon(x, y, alternative="two-sided", mode="exact")
        assert_almost_equal(p, 0.1054688, decimal=6)
        _, p = stats.wilcoxon(x, y, alternative="less", mode="exact")
        assert_almost_equal(p, 0.9580078, decimal=6)
        _, p = stats.wilcoxon(x, y, alternative="greater", mode="exact")
        assert_almost_equal(p, 0.05273438, decimal=6)

        x = np.arange(0, 20) + 0.5
        y = np.arange(20, 0, -1)
        _, p = stats.wilcoxon(x, y, alternative="two-sided", mode="exact")
        assert_almost_equal(p, 0.8694878, decimal=6)
        _, p = stats.wilcoxon(x, y, alternative="less", mode="exact")
        assert_almost_equal(p, 0.4347439, decimal=6)
        _, p = stats.wilcoxon(x, y, alternative="greater", mode="exact")
        assert_almost_equal(p, 0.5795889, decimal=6)

        d = np.arange(26) + 1
        assert_raises(ValueError, stats.wilcoxon, d, mode="exact")

    # These inputs were chosen to give a W statistic that is either the
    # center of the distribution (when the length of the support is odd), or
    # the value to the left of the center (when the length of the support is
    # even).  Also, the numbers are chosen so that the W statistic is the
    # sum of the positive values.
    @pytest.mark.parametrize('x', [[-1, -2, 3],
                                   [-1, 2, -3, -4, 5],
                                   [-1, -2, 3, -4, -5, -6, 7, 8]])
    def test_exact_p_1(self, x):
        w, p = stats.wilcoxon(x)
        x = np.array(x)
        wtrue = x[x > 0].sum()
        assert_equal(w, wtrue)
        assert_equal(p, 1)

    def test_auto(self):
        # auto default to exact if there are no ties and n<= 25
        x = np.arange(0, 25) + 0.5
        y = np.arange(25, 0, -1)
        assert_equal(stats.wilcoxon(x, y),
                     stats.wilcoxon(x, y, mode="exact"))

        # if there are ties (i.e. zeros in d = x-y), then switch to approx
        d = np.arange(0, 13)
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Exact p-value calculation")
            w, p = stats.wilcoxon(d)
        assert_equal(stats.wilcoxon(d, mode="approx"), (w, p))

        # use approximation for samples > 25
        d = np.arange(1, 27)
        assert_equal(stats.wilcoxon(d), stats.wilcoxon(d, mode="approx"))


class TestKstat:
    def test_moments_normal_distribution(self):
        np.random.seed(32149)
        data = np.random.randn(12345)
        moments = [stats.kstat(data, n) for n in [1, 2, 3, 4]]

        expected = [0.011315, 1.017931, 0.05811052, 0.0754134]
        assert_allclose(moments, expected, rtol=1e-4)

        # test equivalence with `stats.moment`
        m1 = stats.moment(data, moment=1)
        m2 = stats.moment(data, moment=2)
        m3 = stats.moment(data, moment=3)
        assert_allclose((m1, m2, m3), expected[:-1], atol=0.02, rtol=1e-2)

    def test_empty_input(self):
        assert_raises(ValueError, stats.kstat, [])

    def test_nan_input(self):
        data = np.arange(10.)
        data[6] = np.nan

        assert_equal(stats.kstat(data), np.nan)

    def test_kstat_bad_arg(self):
        # Raise ValueError if n > 4 or n < 1.
        data = np.arange(10)
        for n in [0, 4.001]:
            assert_raises(ValueError, stats.kstat, data, n=n)


class TestKstatVar:
    def test_empty_input(self):
        assert_raises(ValueError, stats.kstatvar, [])

    def test_nan_input(self):
        data = np.arange(10.)
        data[6] = np.nan

        assert_equal(stats.kstat(data), np.nan)

    def test_bad_arg(self):
        # Raise ValueError is n is not 1 or 2.
        data = [1]
        n = 10
        assert_raises(ValueError, stats.kstatvar, data, n=n)


class TestPpccPlot:
    def setup_method(self):
        self.x = stats.loggamma.rvs(5, size=500, random_state=7654321) + 5

    def test_basic(self):
        N = 5
        svals, ppcc = stats.ppcc_plot(self.x, -10, 10, N=N)
        ppcc_expected = [0.21139644, 0.21384059, 0.98766719, 0.97980182,
                         0.93519298]
        assert_allclose(svals, np.linspace(-10, 10, num=N))
        assert_allclose(ppcc, ppcc_expected)

    def test_dist(self):
        # Test that we can specify distributions both by name and as objects.
        svals1, ppcc1 = stats.ppcc_plot(self.x, -10, 10, dist='tukeylambda')
        svals2, ppcc2 = stats.ppcc_plot(self.x, -10, 10,
                                        dist=stats.tukeylambda)
        assert_allclose(svals1, svals2, rtol=1e-20)
        assert_allclose(ppcc1, ppcc2, rtol=1e-20)
        # Test that 'tukeylambda' is the default dist
        svals3, ppcc3 = stats.ppcc_plot(self.x, -10, 10)
        assert_allclose(svals1, svals3, rtol=1e-20)
        assert_allclose(ppcc1, ppcc3, rtol=1e-20)

    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    def test_plot_kwarg(self):
        # Check with the matplotlib.pyplot module
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.ppcc_plot(self.x, -20, 20, plot=plt)
        fig.delaxes(ax)

        # Check that a Matplotlib Axes object is accepted
        ax = fig.add_subplot(111)
        stats.ppcc_plot(self.x, -20, 20, plot=ax)
        plt.close()

    def test_invalid_inputs(self):
        # `b` has to be larger than `a`
        assert_raises(ValueError, stats.ppcc_plot, self.x, 1, 0)

        # Raise ValueError when given an invalid distribution.
        assert_raises(ValueError, stats.ppcc_plot, [1, 2, 3], 0, 1,
                      dist="plate_of_shrimp")

    def test_empty(self):
        # For consistency with probplot return for one empty array,
        # ppcc contains all zeros and svals is the same as for normal array
        # input.
        svals, ppcc = stats.ppcc_plot([], 0, 1)
        assert_allclose(svals, np.linspace(0, 1, num=80))
        assert_allclose(ppcc, np.zeros(80, dtype=float))


class TestPpccMax:
    def test_ppcc_max_bad_arg(self):
        # Raise ValueError when given an invalid distribution.
        data = [1]
        assert_raises(ValueError, stats.ppcc_max, data, dist="plate_of_shrimp")

    def test_ppcc_max_basic(self):
        x = stats.tukeylambda.rvs(-0.7, loc=2, scale=0.5, size=10000,
                                  random_state=1234567) + 1e4
        assert_almost_equal(stats.ppcc_max(x), -0.71215366521264145, decimal=7)

    def test_dist(self):
        x = stats.tukeylambda.rvs(-0.7, loc=2, scale=0.5, size=10000,
                                  random_state=1234567) + 1e4

        # Test that we can specify distributions both by name and as objects.
        max1 = stats.ppcc_max(x, dist='tukeylambda')
        max2 = stats.ppcc_max(x, dist=stats.tukeylambda)
        assert_almost_equal(max1, -0.71215366521264145, decimal=5)
        assert_almost_equal(max2, -0.71215366521264145, decimal=5)

        # Test that 'tukeylambda' is the default dist
        max3 = stats.ppcc_max(x)
        assert_almost_equal(max3, -0.71215366521264145, decimal=5)

    def test_brack(self):
        x = stats.tukeylambda.rvs(-0.7, loc=2, scale=0.5, size=10000,
                                  random_state=1234567) + 1e4
        assert_raises(ValueError, stats.ppcc_max, x, brack=(0.0, 1.0, 0.5))

        assert_almost_equal(stats.ppcc_max(x, brack=(0, 1)),
                            -0.71215366521264145, decimal=7)

        assert_almost_equal(stats.ppcc_max(x, brack=(-2, 2)),
                            -0.71215366521264145, decimal=7)


class TestBoxcox_llf:

    def test_basic(self):
        x = stats.norm.rvs(size=10000, loc=10, random_state=54321)
        lmbda = 1
        llf = stats.boxcox_llf(lmbda, x)
        llf_expected = -x.size / 2. * np.log(np.sum(x.std()**2))
        assert_allclose(llf, llf_expected)

    def test_array_like(self):
        x = stats.norm.rvs(size=100, loc=10, random_state=54321)
        lmbda = 1
        llf = stats.boxcox_llf(lmbda, x)
        llf2 = stats.boxcox_llf(lmbda, list(x))
        assert_allclose(llf, llf2, rtol=1e-12)

    def test_2d_input(self):
        # Note: boxcox_llf() was already working with 2-D input (sort of), so
        # keep it like that.  boxcox() doesn't work with 2-D input though, due
        # to brent() returning a scalar.
        x = stats.norm.rvs(size=100, loc=10, random_state=54321)
        lmbda = 1
        llf = stats.boxcox_llf(lmbda, x)
        llf2 = stats.boxcox_llf(lmbda, np.vstack([x, x]).T)
        assert_allclose([llf, llf], llf2, rtol=1e-12)

    def test_empty(self):
        assert_(np.isnan(stats.boxcox_llf(1, [])))

    def test_gh_6873(self):
        # Regression test for gh-6873.
        # This example was taken from gh-7534, a duplicate of gh-6873.
        data = [198.0, 233.0, 233.0, 392.0]
        llf = stats.boxcox_llf(-8, data)
        # The expected value was computed with mpmath.
        assert_allclose(llf, -17.93934208579061)


# This is the data from github user Qukaiyi, given as an example
# of a data set that caused boxcox to fail.
_boxcox_data = [
    15957, 112079, 1039553, 711775, 173111, 307382, 183155, 53366, 760875,
    207500, 160045, 473714, 40194, 440319, 133261, 265444, 155590, 36660,
    904939, 55108, 138391, 339146, 458053, 63324, 1377727, 1342632, 41575,
    68685, 172755, 63323, 368161, 199695, 538214, 167760, 388610, 398855,
    1001873, 364591, 1320518, 194060, 194324, 2318551, 196114, 64225, 272000,
    198668, 123585, 86420, 1925556, 695798, 88664, 46199, 759135, 28051,
    345094, 1977752, 51778, 82746, 638126, 2560910, 45830, 140576, 1603787,
    57371, 548730, 5343629, 2298913, 998813, 2156812, 423966, 68350, 145237,
    131935, 1600305, 342359, 111398, 1409144, 281007, 60314, 242004, 113418,
    246211, 61940, 95858, 957805, 40909, 307955, 174159, 124278, 241193,
    872614, 304180, 146719, 64361, 87478, 509360, 167169, 933479, 620561,
    483333, 97416, 143518, 286905, 597837, 2556043, 89065, 69944, 196858,
    88883, 49379, 916265, 1527392, 626954, 54415, 89013, 2883386, 106096,
    402697, 45578, 349852, 140379, 34648, 757343, 1305442, 2054757, 121232,
    606048, 101492, 51426, 1820833, 83412, 136349, 1379924, 505977, 1303486,
    95853, 146451, 285422, 2205423, 259020, 45864, 684547, 182014, 784334,
    174793, 563068, 170745, 1195531, 63337, 71833, 199978, 2330904, 227335,
    898280, 75294, 2011361, 116771, 157489, 807147, 1321443, 1148635, 2456524,
    81839, 1228251, 97488, 1051892, 75397, 3009923, 2732230, 90923, 39735,
    132433, 225033, 337555, 1204092, 686588, 1062402, 40362, 1361829, 1497217,
    150074, 551459, 2019128, 39581, 45349, 1117187, 87845, 1877288, 164448,
    10338362, 24942, 64737, 769946, 2469124, 2366997, 259124, 2667585, 29175,
    56250, 74450, 96697, 5920978, 838375, 225914, 119494, 206004, 430907,
    244083, 219495, 322239, 407426, 618748, 2087536, 2242124, 4736149, 124624,
    406305, 240921, 2675273, 4425340, 821457, 578467, 28040, 348943, 48795,
    145531, 52110, 1645730, 1768364, 348363, 85042, 2673847, 81935, 169075,
    367733, 135474, 383327, 1207018, 93481, 5934183, 352190, 636533, 145870,
    55659, 146215, 73191, 248681, 376907, 1606620, 169381, 81164, 246390,
    236093, 885778, 335969, 49266, 381430, 307437, 350077, 34346, 49340,
    84715, 527120, 40163, 46898, 4609439, 617038, 2239574, 159905, 118337,
    120357, 430778, 3799158, 3516745, 54198, 2970796, 729239, 97848, 6317375,
    887345, 58198, 88111, 867595, 210136, 1572103, 1420760, 574046, 845988,
    509743, 397927, 1119016, 189955, 3883644, 291051, 126467, 1239907, 2556229,
    411058, 657444, 2025234, 1211368, 93151, 577594, 4842264, 1531713, 305084,
    479251, 20591, 1466166, 137417, 897756, 594767, 3606337, 32844, 82426,
    1294831, 57174, 290167, 322066, 813146, 5671804, 4425684, 895607, 450598,
    1048958, 232844, 56871, 46113, 70366, 701618, 97739, 157113, 865047,
    194810, 1501615, 1765727, 38125, 2733376, 40642, 437590, 127337, 106310,
    4167579, 665303, 809250, 1210317, 45750, 1853687, 348954, 156786, 90793,
    1885504, 281501, 3902273, 359546, 797540, 623508, 3672775, 55330, 648221,
    266831, 90030, 7118372, 735521, 1009925, 283901, 806005, 2434897, 94321,
    309571, 4213597, 2213280, 120339, 64403, 8155209, 1686948, 4327743,
    1868312, 135670, 3189615, 1569446, 706058, 58056, 2438625, 520619, 105201,
    141961, 179990, 1351440, 3148662, 2804457, 2760144, 70775, 33807, 1926518,
    2362142, 186761, 240941, 97860, 1040429, 1431035, 78892, 484039, 57845,
    724126, 3166209, 175913, 159211, 1182095, 86734, 1921472, 513546, 326016,
    1891609
]

class TestBoxcox:

    def test_fixed_lmbda(self):
        x = stats.loggamma.rvs(5, size=50, random_state=12345) + 5
        xt = stats.boxcox(x, lmbda=1)
        assert_allclose(xt, x - 1)
        xt = stats.boxcox(x, lmbda=-1)
        assert_allclose(xt, 1 - 1/x)

        xt = stats.boxcox(x, lmbda=0)
        assert_allclose(xt, np.log(x))

        # Also test that array_like input works
        xt = stats.boxcox(list(x), lmbda=0)
        assert_allclose(xt, np.log(x))

    def test_lmbda_None(self):
        # Start from normal rv's, do inverse transform to check that
        # optimization function gets close to the right answer.
        lmbda = 2.5
        x = stats.norm.rvs(loc=10, size=50000, random_state=1245)
        x_inv = (x * lmbda + 1)**(-lmbda)
        xt, maxlog = stats.boxcox(x_inv)

        assert_almost_equal(maxlog, -1 / lmbda, decimal=2)

    def test_alpha(self):
        rng = np.random.RandomState(1234)
        x = stats.loggamma.rvs(5, size=50, random_state=rng) + 5

        # Some regular values for alpha, on a small sample size
        _, _, interval = stats.boxcox(x, alpha=0.75)
        assert_allclose(interval, [4.004485780226041, 5.138756355035744])
        _, _, interval = stats.boxcox(x, alpha=0.05)
        assert_allclose(interval, [1.2138178554857557, 8.209033272375663])

        # Try some extreme values, see we don't hit the N=500 limit
        x = stats.loggamma.rvs(7, size=500, random_state=rng) + 15
        _, _, interval = stats.boxcox(x, alpha=0.001)
        assert_allclose(interval, [0.3988867, 11.40553131])
        _, _, interval = stats.boxcox(x, alpha=0.999)
        assert_allclose(interval, [5.83316246, 5.83735292])

    def test_boxcox_bad_arg(self):
        # Raise ValueError if any data value is negative.
        x = np.array([-1, 2])
        assert_raises(ValueError, stats.boxcox, x)
        # Raise ValueError if data is constant.
        assert_raises(ValueError, stats.boxcox, np.array([1]))
        # Raise ValueError if data is not 1-dimensional.
        assert_raises(ValueError, stats.boxcox, np.array([[1], [2]]))

    def test_empty(self):
        assert_(stats.boxcox([]).shape == (0,))

    def test_gh_6873(self):
        # Regression test for gh-6873.
        y, lam = stats.boxcox(_boxcox_data)
        # The expected value of lam was computed with the function
        # powerTransform in the R library 'car'.  I trust that value
        # to only about five significant digits.
        assert_allclose(lam, -0.051654, rtol=1e-5)

    @pytest.mark.parametrize("bounds", [(-1, 1), (1.1, 2), (-2, -1.1)])
    def test_bounded_optimizer_within_bounds(self, bounds):
        # Define custom optimizer with bounds.
        def optimizer(fun):
            return optimize.minimize_scalar(fun, bounds=bounds,
                                            method="bounded")

        _, lmbda = stats.boxcox(_boxcox_data, lmbda=None, optimizer=optimizer)
        assert bounds[0] < lmbda < bounds[1]

    def test_bounded_optimizer_against_unbounded_optimizer(self):
        # Test whether setting bounds on optimizer excludes solution from
        # unbounded optimizer.

        # Get unbounded solution.
        _, lmbda = stats.boxcox(_boxcox_data, lmbda=None)

        # Set tolerance and bounds around solution.
        bounds = (lmbda + 0.1, lmbda + 1)
        options = {'xatol': 1e-12}

        def optimizer(fun):
            return optimize.minimize_scalar(fun, bounds=bounds,
                                            method="bounded", options=options)

        # Check bounded solution. Lower bound should be active.
        _, lmbda_bounded = stats.boxcox(_boxcox_data, lmbda=None,
                                        optimizer=optimizer)
        assert lmbda_bounded != lmbda
        assert_allclose(lmbda_bounded, bounds[0])

    @pytest.mark.parametrize("optimizer", ["str", (1, 2), 0.1])
    def test_bad_optimizer_type_raises_error(self, optimizer):
        # Check if error is raised if string, tuple or float is passed
        with pytest.raises(ValueError, match="`optimizer` must be a callable"):
            stats.boxcox(_boxcox_data, lmbda=None, optimizer=optimizer)

    def test_bad_optimizer_value_raises_error(self):
        # Check if error is raised if `optimizer` function does not return
        # `OptimizeResult` object

        # Define test function that always returns 1
        def optimizer(fun):
            return 1

        message = "`optimizer` must return an object containing the optimal..."
        with pytest.raises(ValueError, match=message):
            stats.boxcox(_boxcox_data, lmbda=None, optimizer=optimizer)


class TestBoxcoxNormmax:
    def setup_method(self):
        self.x = stats.loggamma.rvs(5, size=50, random_state=12345) + 5

    def test_pearsonr(self):
        maxlog = stats.boxcox_normmax(self.x)
        assert_allclose(maxlog, 1.804465, rtol=1e-6)

    def test_mle(self):
        maxlog = stats.boxcox_normmax(self.x, method='mle')
        assert_allclose(maxlog, 1.758101, rtol=1e-6)

        # Check that boxcox() uses 'mle'
        _, maxlog_boxcox = stats.boxcox(self.x)
        assert_allclose(maxlog_boxcox, maxlog)

    def test_all(self):
        maxlog_all = stats.boxcox_normmax(self.x, method='all')
        assert_allclose(maxlog_all, [1.804465, 1.758101], rtol=1e-6)

    @pytest.mark.parametrize("method", ["mle", "pearsonr", "all"])
    @pytest.mark.parametrize("bounds", [(-1, 1), (1.1, 2), (-2, -1.1)])
    def test_bounded_optimizer_within_bounds(self, method, bounds):

        def optimizer(fun):
            return optimize.minimize_scalar(fun, bounds=bounds,
                                            method="bounded")

        maxlog = stats.boxcox_normmax(self.x, method=method,
                                      optimizer=optimizer)
        assert np.all(bounds[0] < maxlog)
        assert np.all(maxlog < bounds[1])

    def test_user_defined_optimizer(self):
        # tests an optimizer that is not based on scipy.optimize.minimize
        lmbda = stats.boxcox_normmax(self.x)
        lmbda_rounded = np.round(lmbda, 5)
        lmbda_range = np.linspace(lmbda_rounded-0.01, lmbda_rounded+0.01, 1001)

        class MyResult:
            pass

        def optimizer(fun):
            # brute force minimum over the range
            objs = []
            for lmbda in lmbda_range:
                objs.append(fun(lmbda))
            res = MyResult()
            res.x = lmbda_range[np.argmin(objs)]
            return res

        lmbda2 = stats.boxcox_normmax(self.x, optimizer=optimizer)
        assert lmbda2 != lmbda                 # not identical
        assert_allclose(lmbda2, lmbda, 1e-5)   # but as close as it should be

    def test_user_defined_optimizer_and_brack_raises_error(self):
        optimizer = optimize.minimize_scalar

        # Using default `brack=None` with user-defined `optimizer` works as
        # expected.
        stats.boxcox_normmax(self.x, brack=None, optimizer=optimizer)

        # Using user-defined `brack` with user-defined `optimizer` is expected
        # to throw an error. Instead, users should specify
        # optimizer-specific parameters in the optimizer function itself.
        with pytest.raises(ValueError, match="`brack` must be None if "
                                             "`optimizer` is given"):

            stats.boxcox_normmax(self.x, brack=(-2.0, 2.0),
                                 optimizer=optimizer)


class TestBoxcoxNormplot:
    def setup_method(self):
        self.x = stats.loggamma.rvs(5, size=500, random_state=7654321) + 5

    def test_basic(self):
        N = 5
        lmbdas, ppcc = stats.boxcox_normplot(self.x, -10, 10, N=N)
        ppcc_expected = [0.57783375, 0.83610988, 0.97524311, 0.99756057,
                         0.95843297]
        assert_allclose(lmbdas, np.linspace(-10, 10, num=N))
        assert_allclose(ppcc, ppcc_expected)

    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    def test_plot_kwarg(self):
        # Check with the matplotlib.pyplot module
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.boxcox_normplot(self.x, -20, 20, plot=plt)
        fig.delaxes(ax)

        # Check that a Matplotlib Axes object is accepted
        ax = fig.add_subplot(111)
        stats.boxcox_normplot(self.x, -20, 20, plot=ax)
        plt.close()

    def test_invalid_inputs(self):
        # `lb` has to be larger than `la`
        assert_raises(ValueError, stats.boxcox_normplot, self.x, 1, 0)
        # `x` can not contain negative values
        assert_raises(ValueError, stats.boxcox_normplot, [-1, 1], 0, 1)

    def test_empty(self):
        assert_(stats.boxcox_normplot([], 0, 1).size == 0)


class TestYeojohnson_llf:

    def test_array_like(self):
        x = stats.norm.rvs(size=100, loc=0, random_state=54321)
        lmbda = 1
        llf = stats.yeojohnson_llf(lmbda, x)
        llf2 = stats.yeojohnson_llf(lmbda, list(x))
        assert_allclose(llf, llf2, rtol=1e-12)

    def test_2d_input(self):
        x = stats.norm.rvs(size=100, loc=10, random_state=54321)
        lmbda = 1
        llf = stats.yeojohnson_llf(lmbda, x)
        llf2 = stats.yeojohnson_llf(lmbda, np.vstack([x, x]).T)
        assert_allclose([llf, llf], llf2, rtol=1e-12)

    def test_empty(self):
        assert_(np.isnan(stats.yeojohnson_llf(1, [])))


class TestYeojohnson:

    def test_fixed_lmbda(self):
        rng = np.random.RandomState(12345)

        # Test positive input
        x = stats.loggamma.rvs(5, size=50, random_state=rng) + 5
        assert np.all(x > 0)
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt, x)
        xt = stats.yeojohnson(x, lmbda=-1)
        assert_allclose(xt, 1 - 1 / (x + 1))
        xt = stats.yeojohnson(x, lmbda=0)
        assert_allclose(xt, np.log(x + 1))
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt, x)

        # Test negative input
        x = stats.loggamma.rvs(5, size=50, random_state=rng) - 5
        assert np.all(x < 0)
        xt = stats.yeojohnson(x, lmbda=2)
        assert_allclose(xt, -np.log(-x + 1))
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt, x)
        xt = stats.yeojohnson(x, lmbda=3)
        assert_allclose(xt, 1 / (-x + 1) - 1)

        # test both positive and negative input
        x = stats.loggamma.rvs(5, size=50, random_state=rng) - 2
        assert not np.all(x < 0)
        assert not np.all(x >= 0)
        pos = x >= 0
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt[pos], x[pos])
        xt = stats.yeojohnson(x, lmbda=-1)
        assert_allclose(xt[pos], 1 - 1 / (x[pos] + 1))
        xt = stats.yeojohnson(x, lmbda=0)
        assert_allclose(xt[pos], np.log(x[pos] + 1))
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt[pos], x[pos])

        neg = ~pos
        xt = stats.yeojohnson(x, lmbda=2)
        assert_allclose(xt[neg], -np.log(-x[neg] + 1))
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt[neg], x[neg])
        xt = stats.yeojohnson(x, lmbda=3)
        assert_allclose(xt[neg], 1 / (-x[neg] + 1) - 1)

    @pytest.mark.parametrize('lmbda', [0, .1, .5, 2])
    def test_lmbda_None(self, lmbda):
        # Start from normal rv's, do inverse transform to check that
        # optimization function gets close to the right answer.

        def _inverse_transform(x, lmbda):
            x_inv = np.zeros(x.shape, dtype=x.dtype)
            pos = x >= 0

            # when x >= 0
            if abs(lmbda) < np.spacing(1.):
                x_inv[pos] = np.exp(x[pos]) - 1
            else:  # lmbda != 0
                x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1

            # when x < 0
            if abs(lmbda - 2) > np.spacing(1.):
                x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1,
                                           1 / (2 - lmbda))
            else:  # lmbda == 2
                x_inv[~pos] = 1 - np.exp(-x[~pos])

            return x_inv

        n_samples = 20000
        np.random.seed(1234567)
        x = np.random.normal(loc=0, scale=1, size=(n_samples))

        x_inv = _inverse_transform(x, lmbda)
        xt, maxlog = stats.yeojohnson(x_inv)

        assert_allclose(maxlog, lmbda, atol=1e-2)

        assert_almost_equal(0, np.linalg.norm(x - xt) / n_samples, decimal=2)
        assert_almost_equal(0, xt.mean(), decimal=1)
        assert_almost_equal(1, xt.std(), decimal=1)

    def test_empty(self):
        assert_(stats.yeojohnson([]).shape == (0,))

    def test_array_like(self):
        x = stats.norm.rvs(size=100, loc=0, random_state=54321)
        xt1, _ = stats.yeojohnson(x)
        xt2, _ = stats.yeojohnson(list(x))
        assert_allclose(xt1, xt2, rtol=1e-12)

    @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
    def test_input_dtype_complex(self, dtype):
        x = np.arange(6, dtype=dtype)
        err_msg = ('Yeo-Johnson transformation is not defined for complex '
                   'numbers.')
        with pytest.raises(ValueError, match=err_msg):
            stats.yeojohnson(x)

    @pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.int32])
    def test_input_dtype_integer(self, dtype):
        x_int = np.arange(8, dtype=dtype)
        x_float = np.arange(8, dtype=np.float64)
        xt_int, lmbda_int = stats.yeojohnson(x_int)
        xt_float, lmbda_float = stats.yeojohnson(x_float)
        assert_allclose(xt_int, xt_float, rtol=1e-7)
        assert_allclose(lmbda_int, lmbda_float, rtol=1e-7)


class TestYeojohnsonNormmax:
    def setup_method(self):
        self.x = stats.loggamma.rvs(5, size=50, random_state=12345) + 5

    def test_mle(self):
        maxlog = stats.yeojohnson_normmax(self.x)
        assert_allclose(maxlog, 1.876393, rtol=1e-6)

    def test_darwin_example(self):
        # test from original paper "A new family of power transformations to
        # improve normality or symmetry" by Yeo and Johnson.
        x = [6.1, -8.4, 1.0, 2.0, 0.7, 2.9, 3.5, 5.1, 1.8, 3.6, 7.0, 3.0, 9.3,
             7.5, -6.0]
        lmbda = stats.yeojohnson_normmax(x)
        assert np.allclose(lmbda, 1.305, atol=1e-3)


class TestCircFuncs:
    @pytest.mark.parametrize("test_func,expected",
                             [(stats.circmean, 0.167690146),
                              (stats.circvar, 42.51955609),
                              (stats.circstd, 6.520702116)])
    def test_circfuncs(self, test_func, expected):
        x = np.array([355, 5, 2, 359, 10, 350])
        assert_allclose(test_func(x, high=360), expected, rtol=1e-7)

    def test_circfuncs_small(self):
        x = np.array([20, 21, 22, 18, 19, 20.5, 19.2])
        M1 = x.mean()
        M2 = stats.circmean(x, high=360)
        assert_allclose(M2, M1, rtol=1e-5)

        V1 = x.var()
        V2 = stats.circvar(x, high=360)
        assert_allclose(V2, V1, rtol=1e-4)

        S1 = x.std()
        S2 = stats.circstd(x, high=360)
        assert_allclose(S2, S1, rtol=1e-4)

    @pytest.mark.parametrize("test_func, numpy_func",
                             [(stats.circmean, np.mean),
                              (stats.circvar, np.var),
                              (stats.circstd, np.std)])
    def test_circfuncs_close(self, test_func, numpy_func):
        # circfuncs should handle very similar inputs (gh-12740)
        x = np.array([0.12675364631578953] * 10 + [0.12675365920187928] * 100)
        circstat = test_func(x)
        normal = numpy_func(x)
        assert_allclose(circstat, normal, atol=2e-8)

    def test_circmean_axis(self):
        x = np.array([[355, 5, 2, 359, 10, 350],
                      [351, 7, 4, 352, 9, 349],
                      [357, 9, 8, 358, 4, 356]])
        M1 = stats.circmean(x, high=360)
        M2 = stats.circmean(x.ravel(), high=360)
        assert_allclose(M1, M2, rtol=1e-14)

        M1 = stats.circmean(x, high=360, axis=1)
        M2 = [stats.circmean(x[i], high=360) for i in range(x.shape[0])]
        assert_allclose(M1, M2, rtol=1e-14)

        M1 = stats.circmean(x, high=360, axis=0)
        M2 = [stats.circmean(x[:, i], high=360) for i in range(x.shape[1])]
        assert_allclose(M1, M2, rtol=1e-14)

    def test_circvar_axis(self):
        x = np.array([[355, 5, 2, 359, 10, 350],
                      [351, 7, 4, 352, 9, 349],
                      [357, 9, 8, 358, 4, 356]])

        V1 = stats.circvar(x, high=360)
        V2 = stats.circvar(x.ravel(), high=360)
        assert_allclose(V1, V2, rtol=1e-11)

        V1 = stats.circvar(x, high=360, axis=1)
        V2 = [stats.circvar(x[i], high=360) for i in range(x.shape[0])]
        assert_allclose(V1, V2, rtol=1e-11)

        V1 = stats.circvar(x, high=360, axis=0)
        V2 = [stats.circvar(x[:, i], high=360) for i in range(x.shape[1])]
        assert_allclose(V1, V2, rtol=1e-11)

    def test_circstd_axis(self):
        x = np.array([[355, 5, 2, 359, 10, 350],
                      [351, 7, 4, 352, 9, 349],
                      [357, 9, 8, 358, 4, 356]])

        S1 = stats.circstd(x, high=360)
        S2 = stats.circstd(x.ravel(), high=360)
        assert_allclose(S1, S2, rtol=1e-11)

        S1 = stats.circstd(x, high=360, axis=1)
        S2 = [stats.circstd(x[i], high=360) for i in range(x.shape[0])]
        assert_allclose(S1, S2, rtol=1e-11)

        S1 = stats.circstd(x, high=360, axis=0)
        S2 = [stats.circstd(x[:, i], high=360) for i in range(x.shape[1])]
        assert_allclose(S1, S2, rtol=1e-11)

    @pytest.mark.parametrize("test_func,expected",
                             [(stats.circmean, 0.167690146),
                              (stats.circvar, 42.51955609),
                              (stats.circstd, 6.520702116)])
    def test_circfuncs_array_like(self, test_func, expected):
        x = [355, 5, 2, 359, 10, 350]
        assert_allclose(test_func(x, high=360), expected, rtol=1e-7)

    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    def test_empty(self, test_func):
        assert_(np.isnan(test_func([])))

    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    def test_nan_propagate(self, test_func):
        x = [355, 5, 2, 359, 10, 350, np.nan]
        assert_(np.isnan(test_func(x, high=360)))

    @pytest.mark.parametrize("test_func,expected",
                             [(stats.circmean,
                               {None: np.nan, 0: 355.66582264, 1: 0.28725053}),
                              (stats.circvar,
                               {None: np.nan, 0: 16.89976130, 1: 36.51366669}),
                              (stats.circstd,
                               {None: np.nan, 0: 4.11093193, 1: 6.04265394})])
    def test_nan_propagate_array(self, test_func, expected):
        x = np.array([[355, 5, 2, 359, 10, 350, 1],
                      [351, 7, 4, 352, 9, 349, np.nan],
                      [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        for axis in expected.keys():
            out = test_func(x, high=360, axis=axis)
            if axis is None:
                assert_(np.isnan(out))
            else:
                assert_allclose(out[0], expected[axis], rtol=1e-7)
                assert_(np.isnan(out[1:]).all())

    @pytest.mark.parametrize("test_func,expected",
                             [(stats.circmean,
                               {None: 359.4178026893944,
                                0: np.array([353.0, 6.0, 3.0, 355.5, 9.5,
                                             349.5]),
                                1: np.array([0.16769015, 358.66510252])}),
                              (stats.circvar,
                               {None: 55.362093503276725,
                                0: np.array([4.00081258, 1.00005077, 1.00005077,
                                             12.25762620, 0.25000317,
                                             0.25000317]),
                                1: np.array([42.51955609, 67.09872148])}),
                              (stats.circstd,
                               {None: 7.440570778057074,
                                0: np.array([2.00020313, 1.00002539, 1.00002539,
                                             3.50108929, 0.50000317,
                                             0.50000317]),
                                1: np.array([6.52070212, 8.19138093])})])
    def test_nan_omit_array(self, test_func, expected):
        x = np.array([[355, 5, 2, 359, 10, 350, np.nan],
                      [351, 7, 4, 352, 9, 349, np.nan],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        for axis in expected.keys():
            out = test_func(x, high=360, nan_policy='omit', axis=axis)
            if axis is None:
                assert_allclose(out, expected[axis], rtol=1e-7)
            else:
                assert_allclose(out[:-1], expected[axis], rtol=1e-7)
                assert_(np.isnan(out[-1]))

    @pytest.mark.parametrize("test_func,expected",
                             [(stats.circmean, 0.167690146),
                              (stats.circvar, 42.51955609),
                              (stats.circstd, 6.520702116)])
    def test_nan_omit(self, test_func, expected):
        x = [355, 5, 2, 359, 10, 350, np.nan]
        assert_allclose(test_func(x, high=360, nan_policy='omit'),
                        expected, rtol=1e-7)

    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    def test_nan_omit_all(self, test_func):
        x = [np.nan, np.nan, np.nan, np.nan, np.nan]
        assert_(np.isnan(test_func(x, nan_policy='omit')))

    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    def test_nan_omit_all_axis(self, test_func):
        x = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                      [np.nan, np.nan, np.nan, np.nan, np.nan]])
        out = test_func(x, nan_policy='omit', axis=1)
        assert_(np.isnan(out).all())
        assert_(len(out) == 2)

    @pytest.mark.parametrize("x",
                             [[355, 5, 2, 359, 10, 350, np.nan],
                              np.array([[355, 5, 2, 359, 10, 350, np.nan],
                                        [351, 7, 4, 352, np.nan, 9, 349]])])
    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    def test_nan_raise(self, test_func, x):
        assert_raises(ValueError, test_func, x, high=360, nan_policy='raise')

    @pytest.mark.parametrize("x",
                             [[355, 5, 2, 359, 10, 350, np.nan],
                              np.array([[355, 5, 2, 359, 10, 350, np.nan],
                                        [351, 7, 4, 352, np.nan, 9, 349]])])
    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    def test_bad_nan_policy(self, test_func, x):
        assert_raises(ValueError, test_func, x, high=360, nan_policy='foobar')

    def test_circmean_scalar(self):
        x = 1.
        M1 = x
        M2 = stats.circmean(x)
        assert_allclose(M2, M1, rtol=1e-5)

    def test_circmean_range(self):
        # regression test for gh-6420: circmean(..., high, low) must be
        # between `high` and `low`
        m = stats.circmean(np.arange(0, 2, 0.1), np.pi, -np.pi)
        assert_(m < np.pi)
        assert_(m > -np.pi)

    def test_circfuncs_unit8(self):
        # regression test for gh-7255: overflow when working with
        # numpy uint8 data type
        x = np.array([150, 10], dtype='uint8')
        assert_equal(stats.circmean(x, high=180), 170.0)
        assert_allclose(stats.circvar(x, high=180), 437.45871686, rtol=1e-7)
        assert_allclose(stats.circstd(x, high=180), 20.91551378, rtol=1e-7)


class TestMedianTest:

    def test_bad_n_samples(self):
        # median_test requires at least two samples.
        assert_raises(ValueError, stats.median_test, [1, 2, 3])

    def test_empty_sample(self):
        # Each sample must contain at least one value.
        assert_raises(ValueError, stats.median_test, [], [1, 2, 3])

    def test_empty_when_ties_ignored(self):
        # The grand median is 1, and all values in the first argument are
        # equal to the grand median.  With ties="ignore", those values are
        # ignored, which results in the first sample being (in effect) empty.
        # This should raise a ValueError.
        assert_raises(ValueError, stats.median_test,
                      [1, 1, 1, 1], [2, 0, 1], [2, 0], ties="ignore")

    def test_empty_contingency_row(self):
        # The grand median is 1, and with the default ties="below", all the
        # values in the samples are counted as being below the grand median.
        # This would result a row of zeros in the contingency table, which is
        # an error.
        assert_raises(ValueError, stats.median_test, [1, 1, 1], [1, 1, 1])

        # With ties="above", all the values are counted as above the
        # grand median.
        assert_raises(ValueError, stats.median_test, [1, 1, 1], [1, 1, 1],
                      ties="above")

    def test_bad_ties(self):
        assert_raises(ValueError, stats.median_test, [1, 2, 3], [4, 5],
                      ties="foo")

    def test_bad_nan_policy(self):
        assert_raises(ValueError, stats.median_test, [1, 2, 3], [4, 5], nan_policy='foobar')

    def test_bad_keyword(self):
        assert_raises(TypeError, stats.median_test, [1, 2, 3], [4, 5],
                      foo="foo")

    def test_simple(self):
        x = [1, 2, 3]
        y = [1, 2, 3]
        stat, p, med, tbl = stats.median_test(x, y)

        # The median is floating point, but this equality test should be safe.
        assert_equal(med, 2.0)

        assert_array_equal(tbl, [[1, 1], [2, 2]])

        # The expected values of the contingency table equal the contingency
        # table, so the statistic should be 0 and the p-value should be 1.
        assert_equal(stat, 0)
        assert_equal(p, 1)

    def test_ties_options(self):
        # Test the contingency table calculation.
        x = [1, 2, 3, 4]
        y = [5, 6]
        z = [7, 8, 9]
        # grand median is 5.

        # Default 'ties' option is "below".
        stat, p, m, tbl = stats.median_test(x, y, z)
        assert_equal(m, 5)
        assert_equal(tbl, [[0, 1, 3], [4, 1, 0]])

        stat, p, m, tbl = stats.median_test(x, y, z, ties="ignore")
        assert_equal(m, 5)
        assert_equal(tbl, [[0, 1, 3], [4, 0, 0]])

        stat, p, m, tbl = stats.median_test(x, y, z, ties="above")
        assert_equal(m, 5)
        assert_equal(tbl, [[0, 2, 3], [4, 0, 0]])

    def test_nan_policy_options(self):
        x = [1, 2, np.nan]
        y = [4, 5, 6]
        mt1 = stats.median_test(x, y, nan_policy='propagate')
        s, p, m, t = stats.median_test(x, y, nan_policy='omit')

        assert_equal(mt1, (np.nan, np.nan, np.nan, None))
        assert_allclose(s, 0.31250000000000006)
        assert_allclose(p, 0.57615012203057869)
        assert_equal(m, 4.0)
        assert_equal(t, np.array([[0, 2],[2, 1]]))
        assert_raises(ValueError, stats.median_test, x, y, nan_policy='raise')

    def test_basic(self):
        # median_test calls chi2_contingency to compute the test statistic
        # and p-value.  Make sure it hasn't screwed up the call...

        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8]

        stat, p, m, tbl = stats.median_test(x, y)
        assert_equal(m, 4)
        assert_equal(tbl, [[1, 2], [4, 2]])

        exp_stat, exp_p, dof, e = stats.chi2_contingency(tbl)
        assert_allclose(stat, exp_stat)
        assert_allclose(p, exp_p)

        stat, p, m, tbl = stats.median_test(x, y, lambda_=0)
        assert_equal(m, 4)
        assert_equal(tbl, [[1, 2], [4, 2]])

        exp_stat, exp_p, dof, e = stats.chi2_contingency(tbl, lambda_=0)
        assert_allclose(stat, exp_stat)
        assert_allclose(p, exp_p)

        stat, p, m, tbl = stats.median_test(x, y, correction=False)
        assert_equal(m, 4)
        assert_equal(tbl, [[1, 2], [4, 2]])

        exp_stat, exp_p, dof, e = stats.chi2_contingency(tbl, correction=False)
        assert_allclose(stat, exp_stat)
        assert_allclose(p, exp_p)"""
An extension of scipy.stats.stats to support masked arrays

"""
# Original author (2007): Pierre GF Gerard-Marchant


__all__ = ['argstoarray',
           'count_tied_groups',
           'describe',
           'f_oneway', 'find_repeats','friedmanchisquare',
           'kendalltau','kendalltau_seasonal','kruskal','kruskalwallis',
           'ks_twosamp', 'ks_2samp', 'kurtosis', 'kurtosistest',
           'ks_1samp', 'kstest',
           'linregress',
           'mannwhitneyu', 'meppf','mode','moment','mquantiles','msign',
           'normaltest',
           'obrientransform',
           'pearsonr','plotting_positions','pointbiserialr',
           'rankdata',
           'scoreatpercentile','sem',
           'sen_seasonal_slopes','skew','skewtest','spearmanr',
           'siegelslopes', 'theilslopes',
           'tmax','tmean','tmin','trim','trimboth',
           'trimtail','trima','trimr','trimmed_mean','trimmed_std',
           'trimmed_stde','trimmed_var','tsem','ttest_1samp','ttest_onesamp',
           'ttest_ind','ttest_rel','tvar',
           'variation',
           'winsorize',
           'brunnermunzel',
           ]

import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math

import itertools
import warnings
from collections import namedtuple

from . import distributions
import scipy.special as special
import scipy.stats.stats

from ._stats_mstats_common import (
        _find_repeats,
        linregress as stats_linregress,
        LinregressResult as stats_LinregressResult,
        theilslopes as stats_theilslopes,
        siegelslopes as stats_siegelslopes
        )

def _chk_asarray(a, axis):
    # Always returns a masked array, raveled for axis=None
    a = ma.asanyarray(a)
    if axis is None:
        a = ma.ravel(a)
        outaxis = 0
    else:
        outaxis = axis
    return a, outaxis


def _chk2_asarray(a, b, axis):
    a = ma.asanyarray(a)
    b = ma.asanyarray(b)
    if axis is None:
        a = ma.ravel(a)
        b = ma.ravel(b)
        outaxis = 0
    else:
        outaxis = axis
    return a, b, outaxis


def _chk_size(a, b):
    a = ma.asanyarray(a)
    b = ma.asanyarray(b)
    (na, nb) = (a.size, b.size)
    if na != nb:
        raise ValueError("The size of the input array should match!"
                         " (%s <> %s)" % (na, nb))
    return (a, b, na)


def argstoarray(*args):
    """
    Constructs a 2D array from a group of sequences.

    Sequences are filled with missing values to match the length of the longest
    sequence.

    Parameters
    ----------
    args : sequences
        Group of sequences.

    Returns
    -------
    argstoarray : MaskedArray
        A ( `m` x `n` ) masked array, where `m` is the number of arguments and
        `n` the length of the longest argument.

    Notes
    -----
    `numpy.ma.row_stack` has identical behavior, but is called with a sequence
    of sequences.

    Examples
    --------
    A 2D masked array constructed from a group of sequences is returned.

    >>> from scipy.stats.mstats import argstoarray
    >>> argstoarray([1, 2, 3], [4, 5, 6])
    masked_array(
     data=[[1.0, 2.0, 3.0],
           [4.0, 5.0, 6.0]],
     mask=[[False, False, False],
           [False, False, False]],
     fill_value=1e+20)

    The returned masked array filled with missing values when the lengths of
    sequences are different.

    >>> argstoarray([1, 3], [4, 5, 6])
    masked_array(
     data=[[1.0, 3.0, --],
           [4.0, 5.0, 6.0]],
     mask=[[False, False,  True],
           [False, False, False]],
     fill_value=1e+20)

    """
    if len(args) == 1 and not isinstance(args[0], ndarray):
        output = ma.asarray(args[0])
        if output.ndim != 2:
            raise ValueError("The input should be 2D")
    else:
        n = len(args)
        m = max([len(k) for k in args])
        output = ma.array(np.empty((n,m), dtype=float), mask=True)
        for (k,v) in enumerate(args):
            output[k,:len(v)] = v

    output[np.logical_not(np.isfinite(output._data))] = masked
    return output


def find_repeats(arr):
    """Find repeats in arr and return a tuple (repeats, repeat_count).

    The input is cast to float64. Masked values are discarded.

    Parameters
    ----------
    arr : sequence
        Input array. The array is flattened if it is not 1D.

    Returns
    -------
    repeats : ndarray
        Array of repeated values.
    counts : ndarray
        Array of counts.

    """
    # Make sure we get a copy. ma.compressed promises a "new array", but can
    # actually return a reference.
    compr = np.asarray(ma.compressed(arr), dtype=np.float64)
    try:
        need_copy = np.may_share_memory(compr, arr)
    except AttributeError:
        # numpy < 1.8.2 bug: np.may_share_memory([], []) raises,
        # while in numpy 1.8.2 and above it just (correctly) returns False.
        need_copy = False
    if need_copy:
        compr = compr.copy()
    return _find_repeats(compr)


def count_tied_groups(x, use_missing=False):
    """
    Counts the number of tied values.

    Parameters
    ----------
    x : sequence
        Sequence of data on which to counts the ties
    use_missing : bool, optional
        Whether to consider missing values as tied.

    Returns
    -------
    count_tied_groups : dict
        Returns a dictionary (nb of ties: nb of groups).

    Examples
    --------
    >>> from scipy.stats import mstats
    >>> z = [0, 0, 0, 2, 2, 2, 3, 3, 4, 5, 6]
    >>> mstats.count_tied_groups(z)
    {2: 1, 3: 2}

    In the above example, the ties were 0 (3x), 2 (3x) and 3 (2x).

    >>> z = np.ma.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 6])
    >>> mstats.count_tied_groups(z)
    {2: 2, 3: 1}
    >>> z[[1,-1]] = np.ma.masked
    >>> mstats.count_tied_groups(z, use_missing=True)
    {2: 2, 3: 1}

    """
    nmasked = ma.getmask(x).sum()
    # We need the copy as find_repeats will overwrite the initial data
    data = ma.compressed(x).copy()
    (ties, counts) = find_repeats(data)
    nties = {}
    if len(ties):
        nties = dict(zip(np.unique(counts), itertools.repeat(1)))
        nties.update(dict(zip(*find_repeats(counts))))

    if nmasked and use_missing:
        try:
            nties[nmasked] += 1
        except KeyError:
            nties[nmasked] = 1

    return nties


def rankdata(data, axis=None, use_missing=False):
    """Returns the rank (also known as order statistics) of each data point
    along the given axis.

    If some values are tied, their rank is averaged.
    If some values are masked, their rank is set to 0 if use_missing is False,
    or set to the average rank of the unmasked values if use_missing is True.

    Parameters
    ----------
    data : sequence
        Input data. The data is transformed to a masked array
    axis : {None,int}, optional
        Axis along which to perform the ranking.
        If None, the array is first flattened. An exception is raised if
        the axis is specified for arrays with a dimension larger than 2
    use_missing : bool, optional
        Whether the masked values have a rank of 0 (False) or equal to the
        average rank of the unmasked values (True).

    """
    def _rank1d(data, use_missing=False):
        n = data.count()
        rk = np.empty(data.size, dtype=float)
        idx = data.argsort()
        rk[idx[:n]] = np.arange(1,n+1)

        if use_missing:
            rk[idx[n:]] = (n+1)/2.
        else:
            rk[idx[n:]] = 0

        repeats = find_repeats(data.copy())
        for r in repeats[0]:
            condition = (data == r).filled(False)
            rk[condition] = rk[condition].mean()
        return rk

    data = ma.array(data, copy=False)
    if axis is None:
        if data.ndim > 1:
            return _rank1d(data.ravel(), use_missing).reshape(data.shape)
        else:
            return _rank1d(data, use_missing)
    else:
        return ma.apply_along_axis(_rank1d,axis,data,use_missing).view(ndarray)


ModeResult = namedtuple('ModeResult', ('mode', 'count'))


def mode(a, axis=0):
    """
    Returns an array of the modal (most common) value in the passed array.

    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    mode : ndarray
        Array of modal values.
    count : ndarray
        Array of counts for each mode.

    Notes
    -----
    For more details, see `stats.mode`.

    Examples
    --------
    >>> from scipy import stats
    >>> from scipy.stats import mstats
    >>> m_arr = np.ma.array([1, 1, 0, 0, 0, 0], mask=[0, 0, 1, 1, 1, 0])
    >>> stats.mode(m_arr)
    ModeResult(mode=array([0]), count=array([4]))
    >>> mstats.mode(m_arr)
    ModeResult(mode=array([1.]), count=array([2.]))

    """
    a, axis = _chk_asarray(a, axis)

    def _mode1D(a):
        (rep,cnt) = find_repeats(a)
        if not cnt.ndim:
            return (0, 0)
        elif cnt.size:
            return (rep[cnt.argmax()], cnt.max())
        else:
            return (a.min(), 1)

    if axis is None:
        output = _mode1D(ma.ravel(a))
        output = (ma.array(output[0]), ma.array(output[1]))
    else:
        output = ma.apply_along_axis(_mode1D, axis, a)
        newshape = list(a.shape)
        newshape[axis] = 1
        slices = [slice(None)] * output.ndim
        slices[axis] = 0
        modes = output[tuple(slices)].reshape(newshape)
        slices[axis] = 1
        counts = output[tuple(slices)].reshape(newshape)
        output = (modes, counts)

    return ModeResult(*output)


def _betai(a, b, x):
    x = np.asanyarray(x)
    x = ma.where(x < 1.0, x, 1.0)  # if x > 1 then return 1.0
    return special.betainc(a, b, x)


def msign(x):
    """Returns the sign of x, or 0 if x is masked."""
    return ma.filled(np.sign(x), 0)


def pearsonr(x, y):
    """
    Calculates a Pearson correlation coefficient and the p-value for testing
    non-correlation.

    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear
    relationship. Positive correlations imply that as `x` increases, so does
    `y`. Negative correlations imply that as `x` increases, `y` decreases.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.

    Parameters
    ----------
    x : 1-D array_like
        Input
    y : 1-D array_like
        Input

    Returns
    -------
    pearsonr : float
        Pearson's correlation coefficient, 2-tailed p-value.

    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation

    """
    (x, y, n) = _chk_size(x, y)
    (x, y) = (x.ravel(), y.ravel())
    # Get the common mask and the total nb of unmasked elements
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    n -= m.sum()
    df = n-2
    if df < 0:
        return (masked, masked)

    return scipy.stats.stats.pearsonr(ma.masked_array(x, mask=m).compressed(),
                                      ma.masked_array(y, mask=m).compressed())


SpearmanrResult = namedtuple('SpearmanrResult', ('correlation', 'pvalue'))


def spearmanr(x, y=None, use_ties=True, axis=None, nan_policy='propagate',
              alternative='two-sided'):
    """
    Calculates a Spearman rank-order correlation coefficient and the p-value
    to test for non-correlation.

    The Spearman correlation is a nonparametric measure of the linear
    relationship between two datasets. Unlike the Pearson correlation, the
    Spearman correlation does not assume that both datasets are normally
    distributed. Like other correlation coefficients, this one varies
    between -1 and +1 with 0 implying no correlation. Correlations of -1 or
    +1 imply a monotonic relationship. Positive correlations imply that
    as `x` increases, so does `y`. Negative correlations imply that as `x`
    increases, `y` decreases.

    Missing values are discarded pair-wise: if a value is missing in `x`, the
    corresponding value in `y` is masked.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Spearman correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.

    Parameters
    ----------
    x, y : 1D or 2D array_like, y is optional
        One or two 1-D or 2-D arrays containing multiple variables and
        observations. When these are 1-D, each represents a vector of
        observations of a single variable. For the behavior in the 2-D case,
        see under ``axis``, below.
    use_ties : bool, optional
        DO NOT USE.  Does not do anything, keyword is only left in place for
        backwards compatibility reasons.
    axis : int or None, optional
        If axis=0 (default), then each column represents a variable, with
        observations in the rows. If axis=1, the relationship is transposed:
        each row represents a variable, while the columns contain observations.
        If axis=None, then both arrays will be raveled.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the correlation is nonzero
        * 'less': the correlation is negative (less than zero)
        * 'greater':  the correlation is positive (greater than zero)

        .. versionadded:: 1.7.0

    Returns
    -------
    correlation : float
        Spearman correlation coefficient
    pvalue : float
        2-tailed p-value.

    References
    ----------
    [CRCProbStat2000] section 14.7

    """
    if not use_ties:
        raise ValueError("`use_ties=False` is not supported in SciPy >= 1.2.0")

    # Always returns a masked array, raveled if axis=None
    x, axisout = _chk_asarray(x, axis)
    if y is not None:
        # Deal only with 2-D `x` case.
        y, _ = _chk_asarray(y, axis)
        if axisout == 0:
            x = ma.column_stack((x, y))
        else:
            x = ma.row_stack((x, y))

    if axisout == 1:
        # To simplify the code that follow (always use `n_obs, n_vars` shape)
        x = x.T

    if nan_policy == 'omit':
        x = ma.masked_invalid(x)

    def _spearmanr_2cols(x):
        # Mask the same observations for all variables, and then drop those
        # observations (can't leave them masked, rankdata is weird).
        x = ma.mask_rowcols(x, axis=0)
        x = x[~x.mask.any(axis=1), :]

        # If either column is entirely NaN or Inf
        if not np.any(x.data):
            return SpearmanrResult(np.nan, np.nan)

        m = ma.getmask(x)
        n_obs = x.shape[0]
        dof = n_obs - 2 - int(m.sum(axis=0)[0])
        if dof < 0:
            raise ValueError("The input must have at least 3 entries!")

        # Gets the ranks and rank differences
        x_ranked = rankdata(x, axis=0)
        rs = ma.corrcoef(x_ranked, rowvar=False).data

        # rs can have elements equal to 1, so avoid zero division warnings
        with np.errstate(divide='ignore'):
            # clip the small negative values possibly caused by rounding
            # errors before taking the square root
            t = rs * np.sqrt((dof / ((rs+1.0) * (1.0-rs))).clip(0))

        t, prob = scipy.stats.stats._ttest_finish(dof, t, alternative)

        # For backwards compatibility, return scalars when comparing 2 columns
        if rs.shape == (2, 2):
            return SpearmanrResult(rs[1, 0], prob[1, 0])
        else:
            return SpearmanrResult(rs, prob)

    # Need to do this per pair of variables, otherwise the dropped observations
    # in a third column mess up the result for a pair.
    n_vars = x.shape[1]
    if n_vars == 2:
        return _spearmanr_2cols(x)
    else:
        rs = np.ones((n_vars, n_vars), dtype=float)
        prob = np.zeros((n_vars, n_vars), dtype=float)
        for var1 in range(n_vars - 1):
            for var2 in range(var1+1, n_vars):
                result = _spearmanr_2cols(x[:, [var1, var2]])
                rs[var1, var2] = result.correlation
                rs[var2, var1] = result.correlation
                prob[var1, var2] = result.pvalue
                prob[var2, var1] = result.pvalue

        return SpearmanrResult(rs, prob)


def _kendall_p_exact(n, c):
    # Exact p-value, see Maurice G. Kendall, "Rank Correlation Methods" (4th Edition), Charles Griffin & Co., 1970.
    if n <= 0:
        raise ValueError(f'n ({n}) must be positive')
    elif c < 0 or 4*c > n*(n-1):
        raise ValueError(f'c ({c}) must satisfy 0 <= 4c <= n(n-1) = {n*(n-1)}.')
    elif n == 1:
        prob = 1.0
    elif n == 2:
        prob = 1.0
    elif c == 0:
        prob = 2.0/math.factorial(n) if n < 171 else 0.0
    elif c == 1:
        prob = 2.0/math.factorial(n-1) if n < 172 else 0.0
    elif 4*c == n*(n-1):
        prob = 1.0
    elif n < 171:
        new = np.zeros(c+1)
        new[0:2] = 1.0
        for j in range(3,n+1):
            new = np.cumsum(new)
            if j <= c:
                new[j:] -= new[:c+1-j]
        prob = 2.0*np.sum(new)/math.factorial(n)
    else:
        new = np.zeros(c+1)
        new[0:2] = 1.0
        for j in range(3, n+1):
            new = np.cumsum(new)/j
            if j <= c:
                new[j:] -= new[:c+1-j]
        prob = np.sum(new)

    return np.clip(prob, 0, 1)


KendalltauResult = namedtuple('KendalltauResult', ('correlation', 'pvalue'))


def kendalltau(x, y, use_ties=True, use_missing=False, method='auto'):
    """
    Computes Kendall's rank correlation tau on two variables *x* and *y*.

    Parameters
    ----------
    x : sequence
        First data list (for example, time).
    y : sequence
        Second data list.
    use_ties : {True, False}, optional
        Whether ties correction should be performed.
    use_missing : {False, True}, optional
        Whether missing data should be allocated a rank of 0 (False) or the
        average rank (True)
    method: {'auto', 'asymptotic', 'exact'}, optional
        Defines which method is used to calculate the p-value [1]_.
        'asymptotic' uses a normal approximation valid for large samples.
        'exact' computes the exact p-value, but can only be used if no ties
        are present. As the sample size increases, the 'exact' computation
        time may grow and the result may lose some precision.
        'auto' is the default and selects the appropriate
        method based on a trade-off between speed and accuracy.

    Returns
    -------
    correlation : float
        Kendall tau
    pvalue : float
        Approximate 2-side p-value.

    References
    ----------
    .. [1] Maurice G. Kendall, "Rank Correlation Methods" (4th Edition),
           Charles Griffin & Co., 1970.

    """
    (x, y, n) = _chk_size(x, y)
    (x, y) = (x.flatten(), y.flatten())
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    if m is not nomask:
        x = ma.array(x, mask=m, copy=True)
        y = ma.array(y, mask=m, copy=True)
        # need int() here, otherwise numpy defaults to 32 bit
        # integer on all Windows architectures, causing overflow.
        # int() will keep it infinite precision.
        n -= int(m.sum())

    if n < 2:
        return KendalltauResult(np.nan, np.nan)

    rx = ma.masked_equal(rankdata(x, use_missing=use_missing), 0)
    ry = ma.masked_equal(rankdata(y, use_missing=use_missing), 0)
    idx = rx.argsort()
    (rx, ry) = (rx[idx], ry[idx])
    C = np.sum([((ry[i+1:] > ry[i]) * (rx[i+1:] > rx[i])).filled(0).sum()
                for i in range(len(ry)-1)], dtype=float)
    D = np.sum([((ry[i+1:] < ry[i])*(rx[i+1:] > rx[i])).filled(0).sum()
                for i in range(len(ry)-1)], dtype=float)
    xties = count_tied_groups(x)
    yties = count_tied_groups(y)
    if use_ties:
        corr_x = np.sum([v*k*(k-1) for (k,v) in xties.items()], dtype=float)
        corr_y = np.sum([v*k*(k-1) for (k,v) in yties.items()], dtype=float)
        denom = ma.sqrt((n*(n-1)-corr_x)/2. * (n*(n-1)-corr_y)/2.)
    else:
        denom = n*(n-1)/2.
    tau = (C-D) / denom

    if method == 'exact' and (xties or yties):
        raise ValueError("Ties found, exact method cannot be used.")

    if method == 'auto':
        if (not xties and not yties) and (n <= 33 or min(C, n*(n-1)/2.0-C) <= 1):
            method = 'exact'
        else:
            method = 'asymptotic'

    if not xties and not yties and method == 'exact':
        prob = _kendall_p_exact(n, int(min(C, (n*(n-1))//2-C)))

    elif method == 'asymptotic':
        var_s = n*(n-1)*(2*n+5)
        if use_ties:
            var_s -= np.sum([v*k*(k-1)*(2*k+5)*1. for (k,v) in xties.items()])
            var_s -= np.sum([v*k*(k-1)*(2*k+5)*1. for (k,v) in yties.items()])
            v1 = np.sum([v*k*(k-1) for (k, v) in xties.items()], dtype=float) *\
                 np.sum([v*k*(k-1) for (k, v) in yties.items()], dtype=float)
            v1 /= 2.*n*(n-1)
            if n > 2:
                v2 = np.sum([v*k*(k-1)*(k-2) for (k,v) in xties.items()],
                            dtype=float) * \
                     np.sum([v*k*(k-1)*(k-2) for (k,v) in yties.items()],
                            dtype=float)
                v2 /= 9.*n*(n-1)*(n-2)
            else:
                v2 = 0
        else:
            v1 = v2 = 0

        var_s /= 18.
        var_s += (v1 + v2)
        z = (C-D)/np.sqrt(var_s)
        prob = special.erfc(abs(z)/np.sqrt(2))
    else:
        raise ValueError("Unknown method "+str(method)+" specified, please "
                         "use auto, exact or asymptotic.")

    return KendalltauResult(tau, prob)


def kendalltau_seasonal(x):
    """
    Computes a multivariate Kendall's rank correlation tau, for seasonal data.

    Parameters
    ----------
    x : 2-D ndarray
        Array of seasonal data, with seasons in columns.

    """
    x = ma.array(x, subok=True, copy=False, ndmin=2)
    (n,m) = x.shape
    n_p = x.count(0)

    S_szn = sum(msign(x[i:]-x[i]).sum(0) for i in range(n))
    S_tot = S_szn.sum()

    n_tot = x.count()
    ties = count_tied_groups(x.compressed())
    corr_ties = sum(v*k*(k-1) for (k,v) in ties.items())
    denom_tot = ma.sqrt(1.*n_tot*(n_tot-1)*(n_tot*(n_tot-1)-corr_ties))/2.

    R = rankdata(x, axis=0, use_missing=True)
    K = ma.empty((m,m), dtype=int)
    covmat = ma.empty((m,m), dtype=float)
    denom_szn = ma.empty(m, dtype=float)
    for j in range(m):
        ties_j = count_tied_groups(x[:,j].compressed())
        corr_j = sum(v*k*(k-1) for (k,v) in ties_j.items())
        cmb = n_p[j]*(n_p[j]-1)
        for k in range(j,m,1):
            K[j,k] = sum(msign((x[i:,j]-x[i,j])*(x[i:,k]-x[i,k])).sum()
                               for i in range(n))
            covmat[j,k] = (K[j,k] + 4*(R[:,j]*R[:,k]).sum() -
                           n*(n_p[j]+1)*(n_p[k]+1))/3.
            K[k,j] = K[j,k]
            covmat[k,j] = covmat[j,k]

        denom_szn[j] = ma.sqrt(cmb*(cmb-corr_j)) / 2.

    var_szn = covmat.diagonal()

    z_szn = msign(S_szn) * (abs(S_szn)-1) / ma.sqrt(var_szn)
    z_tot_ind = msign(S_tot) * (abs(S_tot)-1) / ma.sqrt(var_szn.sum())
    z_tot_dep = msign(S_tot) * (abs(S_tot)-1) / ma.sqrt(covmat.sum())

    prob_szn = special.erfc(abs(z_szn)/np.sqrt(2))
    prob_tot_ind = special.erfc(abs(z_tot_ind)/np.sqrt(2))
    prob_tot_dep = special.erfc(abs(z_tot_dep)/np.sqrt(2))

    chi2_tot = (z_szn*z_szn).sum()
    chi2_trd = m * z_szn.mean()**2
    output = {'seasonal tau': S_szn/denom_szn,
              'global tau': S_tot/denom_tot,
              'global tau (alt)': S_tot/denom_szn.sum(),
              'seasonal p-value': prob_szn,
              'global p-value (indep)': prob_tot_ind,
              'global p-value (dep)': prob_tot_dep,
              'chi2 total': chi2_tot,
              'chi2 trend': chi2_trd,
              }
    return output


PointbiserialrResult = namedtuple('PointbiserialrResult', ('correlation',
                                                           'pvalue'))


def pointbiserialr(x, y):
    """Calculates a point biserial correlation coefficient and its p-value.

    Parameters
    ----------
    x : array_like of bools
        Input array.
    y : array_like
        Input array.

    Returns
    -------
    correlation : float
        R value
    pvalue : float
        2-tailed p-value

    Notes
    -----
    Missing values are considered pair-wise: if a value is missing in x,
    the corresponding value in y is masked.

    For more details on `pointbiserialr`, see `stats.pointbiserialr`.

    """
    x = ma.fix_invalid(x, copy=True).astype(bool)
    y = ma.fix_invalid(y, copy=True).astype(float)
    # Get rid of the missing data
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    if m is not nomask:
        unmask = np.logical_not(m)
        x = x[unmask]
        y = y[unmask]

    n = len(x)
    # phat is the fraction of x values that are True
    phat = x.sum() / float(n)
    y0 = y[~x]  # y-values where x is False
    y1 = y[x]  # y-values where x is True
    y0m = y0.mean()
    y1m = y1.mean()

    rpb = (y1m - y0m)*np.sqrt(phat * (1-phat)) / y.std()

    df = n-2
    t = rpb*ma.sqrt(df/(1.0-rpb**2))
    prob = _betai(0.5*df, 0.5, df/(df+t*t))

    return PointbiserialrResult(rpb, prob)


def linregress(x, y=None):
    r"""
    Linear regression calculation

    Note that the non-masked version is used, and that this docstring is
    replaced by the non-masked docstring + some info on missing data.

    """
    if y is None:
        x = ma.array(x)
        if x.shape[0] == 2:
            x, y = x
        elif x.shape[1] == 2:
            x, y = x.T
        else:
            raise ValueError("If only `x` is given as input, "
                             "it has to be of shape (2, N) or (N, 2), "
                             f"provided shape was {x.shape}")
    else:
        x = ma.array(x)
        y = ma.array(y)

    x = x.flatten()
    y = y.flatten()

    m = ma.mask_or(ma.getmask(x), ma.getmask(y), shrink=False)
    if m is not nomask:
        x = ma.array(x, mask=m)
        y = ma.array(y, mask=m)
        if np.any(~m):
            result = stats_linregress(x.data[~m], y.data[~m])
        else:
            # All data is masked
            result = stats_LinregressResult(slope=None, intercept=None,
                                            rvalue=None, pvalue=None,
                                            stderr=None,
                                            intercept_stderr=None)
    else:
        result = stats_linregress(x.data, y.data)

    return result


def theilslopes(y, x=None, alpha=0.95):
    r"""
    Computes the Theil-Sen estimator for a set of points (x, y).

    `theilslopes` implements a method for robust linear regression.  It
    computes the slope as the median of all slopes between paired values.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    x : array_like or None, optional
        Independent variable. If None, use ``arange(len(y))`` instead.
    alpha : float, optional
        Confidence degree between 0 and 1. Default is 95% confidence.
        Note that `alpha` is symmetric around 0.5, i.e. both 0.1 and 0.9 are
        interpreted as "find the 90% confidence interval".

    Returns
    -------
    medslope : float
        Theil slope.
    medintercept : float
        Intercept of the Theil line, as ``median(y) - medslope*median(x)``.
    lo_slope : float
        Lower bound of the confidence interval on `medslope`.
    up_slope : float
        Upper bound of the confidence interval on `medslope`.

    See also
    --------
    siegelslopes : a similar technique with repeated medians


    Notes
    -----
    For more details on `theilslopes`, see `stats.theilslopes`.

    """
    y = ma.asarray(y).flatten()
    if x is None:
        x = ma.arange(len(y), dtype=float)
    else:
        x = ma.asarray(x).flatten()
        if len(x) != len(y):
            raise ValueError("Incompatible lengths ! (%s<>%s)" % (len(y),len(x)))

    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    y._mask = x._mask = m
    # Disregard any masked elements of x or y
    y = y.compressed()
    x = x.compressed().astype(float)
    # We now have unmasked arrays so can use `stats.theilslopes`
    return stats_theilslopes(y, x, alpha=alpha)


def siegelslopes(y, x=None, method="hierarchical"):
    r"""
    Computes the Siegel estimator for a set of points (x, y).

    `siegelslopes` implements a method for robust linear regression
    using repeated medians to fit a line to the points (x, y).
    The method is robust to outliers with an asymptotic breakdown point
    of 50%.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    x : array_like or None, optional
        Independent variable. If None, use ``arange(len(y))`` instead.
    method : {'hierarchical', 'separate'}
        If 'hierarchical', estimate the intercept using the estimated
        slope ``medslope`` (default option).
        If 'separate', estimate the intercept independent of the estimated
        slope. See Notes for details.

    Returns
    -------
    medslope : float
        Estimate of the slope of the regression line.
    medintercept : float
        Estimate of the intercept of the regression line.

    See also
    --------
    theilslopes : a similar technique without repeated medians

    Notes
    -----
    For more details on `siegelslopes`, see `scipy.stats.siegelslopes`.

    """
    y = ma.asarray(y).ravel()
    if x is None:
        x = ma.arange(len(y), dtype=float)
    else:
        x = ma.asarray(x).ravel()
        if len(x) != len(y):
            raise ValueError("Incompatible lengths ! (%s<>%s)" % (len(y), len(x)))

    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    y._mask = x._mask = m
    # Disregard any masked elements of x or y
    y = y.compressed()
    x = x.compressed().astype(float)
    # We now have unmasked arrays so can use `stats.siegelslopes`
    return stats_siegelslopes(y, x)


def sen_seasonal_slopes(x):
    x = ma.array(x, subok=True, copy=False, ndmin=2)
    (n,_) = x.shape
    # Get list of slopes per season
    szn_slopes = ma.vstack([(x[i+1:]-x[i])/np.arange(1,n-i)[:,None]
                            for i in range(n)])
    szn_medslopes = ma.median(szn_slopes, axis=0)
    medslope = ma.median(szn_slopes, axis=None)
    return szn_medslopes, medslope


Ttest_1sampResult = namedtuple('Ttest_1sampResult', ('statistic', 'pvalue'))


def ttest_1samp(a, popmean, axis=0):
    """
    Calculates the T-test for the mean of ONE group of scores.

    Parameters
    ----------
    a : array_like
        sample observation
    popmean : float or array_like
        expected value in null hypothesis, if array_like than it must have the
        same shape as `a` excluding the axis dimension
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        array `a`.

    Returns
    -------
    statistic : float or array
        t-statistic
    pvalue : float or array
        two-tailed p-value

    Notes
    -----
    For more details on `ttest_1samp`, see `stats.ttest_1samp`.

    """
    a, axis = _chk_asarray(a, axis)
    if a.size == 0:
        return (np.nan, np.nan)

    x = a.mean(axis=axis)
    v = a.var(axis=axis, ddof=1)
    n = a.count(axis=axis)
    # force df to be an array for masked division not to throw a warning
    df = ma.asanyarray(n - 1.0)
    svar = ((n - 1.0) * v) / df
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (x - popmean) / ma.sqrt(svar / n)
    prob = special.betainc(0.5*df, 0.5, df/(df + t*t))

    return Ttest_1sampResult(t, prob)


ttest_onesamp = ttest_1samp


Ttest_indResult = namedtuple('Ttest_indResult', ('statistic', 'pvalue'))


def ttest_ind(a, b, axis=0, equal_var=True):
    """
    Calculates the T-test for the means of TWO INDEPENDENT samples of scores.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        arrays, `a`, and `b`.
    equal_var : bool, optional
        If True, perform a standard independent 2 sample test that assumes equal
        population variances.
        If False, perform Welch's t-test, which does not assume equal population
        variance.

        .. versionadded:: 0.17.0

    Returns
    -------
    statistic : float or array
        The calculated t-statistic.
    pvalue : float or array
        The two-tailed p-value.

    Notes
    -----
    For more details on `ttest_ind`, see `stats.ttest_ind`.

    """
    a, b, axis = _chk2_asarray(a, b, axis)

    if a.size == 0 or b.size == 0:
        return Ttest_indResult(np.nan, np.nan)

    (x1, x2) = (a.mean(axis), b.mean(axis))
    (v1, v2) = (a.var(axis=axis, ddof=1), b.var(axis=axis, ddof=1))
    (n1, n2) = (a.count(axis), b.count(axis))

    if equal_var:
        # force df to be an array for masked division not to throw a warning
        df = ma.asanyarray(n1 + n2 - 2.0)
        svar = ((n1-1)*v1+(n2-1)*v2) / df
        denom = ma.sqrt(svar*(1.0/n1 + 1.0/n2))  # n-D computation here!
    else:
        vn1 = v1/n1
        vn2 = v2/n2
        with np.errstate(divide='ignore', invalid='ignore'):
            df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

        # If df is undefined, variances are zero.
        # It doesn't matter what df is as long as it is not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = ma.sqrt(vn1 + vn2)

    with np.errstate(divide='ignore', invalid='ignore'):
        t = (x1-x2) / denom
    probs = special.betainc(0.5*df, 0.5, df/(df + t*t)).reshape(t.shape)

    return Ttest_indResult(t, probs.squeeze())


Ttest_relResult = namedtuple('Ttest_relResult', ('statistic', 'pvalue'))


def ttest_rel(a, b, axis=0):
    """
    Calculates the T-test on TWO RELATED samples of scores, a and b.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape.
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        arrays, `a`, and `b`.

    Returns
    -------
    statistic : float or array
        t-statistic
    pvalue : float or array
        two-tailed p-value

    Notes
    -----
    For more details on `ttest_rel`, see `stats.ttest_rel`.

    """
    a, b, axis = _chk2_asarray(a, b, axis)
    if len(a) != len(b):
        raise ValueError('unequal length arrays')

    if a.size == 0 or b.size == 0:
        return Ttest_relResult(np.nan, np.nan)

    n = a.count(axis)
    df = ma.asanyarray(n-1.0)
    d = (a-b).astype('d')
    dm = d.mean(axis)
    v = d.var(axis=axis, ddof=1)
    denom = ma.sqrt(v / n)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = dm / denom

    probs = special.betainc(0.5*df, 0.5, df/(df + t*t)).reshape(t.shape).squeeze()

    return Ttest_relResult(t, probs)


MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic',
                                                       'pvalue'))


def mannwhitneyu(x,y, use_continuity=True):
    """
    Computes the Mann-Whitney statistic

    Missing values in `x` and/or `y` are discarded.

    Parameters
    ----------
    x : sequence
        Input
    y : sequence
        Input
    use_continuity : {True, False}, optional
        Whether a continuity correction (1/2.) should be taken into account.

    Returns
    -------
    statistic : float
        The minimum of the Mann-Whitney statistics
    pvalue : float
        Approximate two-sided p-value assuming a normal distribution.

    """
    x = ma.asarray(x).compressed().view(ndarray)
    y = ma.asarray(y).compressed().view(ndarray)
    ranks = rankdata(np.concatenate([x,y]))
    (nx, ny) = (len(x), len(y))
    nt = nx + ny
    U = ranks[:nx].sum() - nx*(nx+1)/2.
    U = max(U, nx*ny - U)
    u = nx*ny - U

    mu = (nx*ny)/2.
    sigsq = (nt**3 - nt)/12.
    ties = count_tied_groups(ranks)
    sigsq -= sum(v*(k**3-k) for (k,v) in ties.items())/12.
    sigsq *= nx*ny/float(nt*(nt-1))

    if use_continuity:
        z = (U - 1/2. - mu) / ma.sqrt(sigsq)
    else:
        z = (U - mu) / ma.sqrt(sigsq)

    prob = special.erfc(abs(z)/np.sqrt(2))
    return MannwhitneyuResult(u, prob)


KruskalResult = namedtuple('KruskalResult', ('statistic', 'pvalue'))


def kruskal(*args):
    """
    Compute the Kruskal-Wallis H-test for independent samples

    Parameters
    ----------
    sample1, sample2, ... : array_like
       Two or more arrays with the sample measurements can be given as
       arguments.

    Returns
    -------
    statistic : float
       The Kruskal-Wallis H statistic, corrected for ties
    pvalue : float
       The p-value for the test using the assumption that H has a chi
       square distribution

    Notes
    -----
    For more details on `kruskal`, see `stats.kruskal`.

    Examples
    --------
    >>> from scipy.stats.mstats import kruskal

    Random samples from three different brands of batteries were tested
    to see how long the charge lasted. Results were as follows:

    >>> a = [6.3, 5.4, 5.7, 5.2, 5.0]
    >>> b = [6.9, 7.0, 6.1, 7.9]
    >>> c = [7.2, 6.9, 6.1, 6.5]

    Test the hypotesis that the distribution functions for all of the brands'
    durations are identical. Use 5% level of significance.

    >>> kruskal(a, b, c)
    KruskalResult(statistic=7.113812154696133, pvalue=0.028526948491942164)

    The null hypothesis is rejected at the 5% level of significance
    because the returned p-value is less than the critical value of 5%.

    """
    output = argstoarray(*args)
    ranks = ma.masked_equal(rankdata(output, use_missing=False), 0)
    sumrk = ranks.sum(-1)
    ngrp = ranks.count(-1)
    ntot = ranks.count()
    H = 12./(ntot*(ntot+1)) * (sumrk**2/ngrp).sum() - 3*(ntot+1)
    # Tie correction
    ties = count_tied_groups(ranks)
    T = 1. - sum(v*(k**3-k) for (k,v) in ties.items())/float(ntot**3-ntot)
    if T == 0:
        raise ValueError('All numbers are identical in kruskal')

    H /= T
    df = len(output) - 1
    prob = distributions.chi2.sf(H, df)
    return KruskalResult(H, prob)


kruskalwallis = kruskal


def ks_1samp(x, cdf, args=(), alternative="two-sided", mode='auto'):
    """
    Computes the Kolmogorov-Smirnov test on one sample of masked values.

    Missing values in `x` are discarded.

    Parameters
    ----------
    x : array_like
        a 1-D array of observations of random variables.
    cdf : str or callable
        If a string, it should be the name of a distribution in `scipy.stats`.
        If a callable, that callable is used to calculate the cdf.
    args : tuple, sequence, optional
        Distribution parameters, used if `cdf` is a string.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Indicates the alternative hypothesis.  Default is 'two-sided'.
    mode : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):

          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use approximation to exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic

    Returns
    -------
    d : float
        Value of the Kolmogorov Smirnov test
    p : float
        Corresponding p-value.

    """
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
       alternative.lower()[0], alternative)
    return scipy.stats.stats.ks_1samp(
        x, cdf, args=args, alternative=alternative, mode=mode)


def ks_2samp(data1, data2, alternative="two-sided", mode='auto'):
    """
    Computes the Kolmogorov-Smirnov test on two samples.

    Missing values in `x` and/or `y` are discarded.

    Parameters
    ----------
    data1 : array_like
        First data set
    data2 : array_like
        Second data set
    alternative : {'two-sided', 'less', 'greater'}, optional
        Indicates the alternative hypothesis.  Default is 'two-sided'.
    mode : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):

          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use approximation to exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic

    Returns
    -------
    d : float
        Value of the Kolmogorov Smirnov test
    p : float
        Corresponding p-value.

    """
    # Ideally this would be accomplished by
    # ks_2samp = scipy.stats.stats.ks_2samp
    # but the circular dependencies between mstats_basic and stats prevent that.
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
       alternative.lower()[0], alternative)
    return scipy.stats.stats.ks_2samp(data1, data2, alternative=alternative,
                                      mode=mode)


ks_twosamp = ks_2samp


def kstest(data1, data2, args=(), alternative='two-sided', mode='auto'):
    """

    Parameters
    ----------
    data1 : array_like
    data2 : str, callable or array_like
    args : tuple, sequence, optional
        Distribution parameters, used if `data1` or `data2` are strings.
    alternative : str, as documented in stats.kstest
    mode : str, as documented in stats.kstest

    Returns
    -------
    tuple of (K-S statistic, probability)

    """
    return scipy.stats.stats.kstest(data1, data2, args,
                                    alternative=alternative, mode=mode)


def trima(a, limits=None, inclusive=(True,True)):
    """
    Trims an array by masking the data outside some given limits.

    Returns a masked version of the input array.

    Parameters
    ----------
    a : array_like
        Input array.
    limits : {None, tuple}, optional
        Tuple of (lower limit, upper limit) in absolute values.
        Values of the input array lower (greater) than the lower (upper) limit
        will be masked.  A limit is None indicates an open interval.
    inclusive : (bool, bool) tuple, optional
        Tuple of (lower flag, upper flag), indicating whether values exactly
        equal to the lower (upper) limit are allowed.

    Examples
    --------
    >>> from scipy.stats.mstats import trima

    >>> a = np.arange(10)

    The interval is left-closed and right-open, i.e., `[2, 8)`.
    Trim the array by keeping only values in the interval.

    >>> trima(a, limits=(2, 8), inclusive=(True, False))
    masked_array(data=[--, --, 2, 3, 4, 5, 6, 7, --, --],
                 mask=[ True,  True, False, False, False, False, False, False,
                        True,  True],
           fill_value=999999)

    """
    a = ma.asarray(a)
    a.unshare_mask()
    if (limits is None) or (limits == (None, None)):
        return a

    (lower_lim, upper_lim) = limits
    (lower_in, upper_in) = inclusive
    condition = False
    if lower_lim is not None:
        if lower_in:
            condition |= (a < lower_lim)
        else:
            condition |= (a <= lower_lim)

    if upper_lim is not None:
        if upper_in:
            condition |= (a > upper_lim)
        else:
            condition |= (a >= upper_lim)

    a[condition.filled(True)] = masked
    return a


def trimr(a, limits=None, inclusive=(True, True), axis=None):
    """
    Trims an array by masking some proportion of the data on each end.
    Returns a masked version of the input array.

    Parameters
    ----------
    a : sequence
        Input array.
    limits : {None, tuple}, optional
        Tuple of the percentages to cut on each side of the array, with respect
        to the number of unmasked data, as floats between 0. and 1.
        Noting n the number of unmasked data before trimming, the
        (n*limits[0])th smallest data and the (n*limits[1])th largest data are
        masked, and the total number of unmasked data after trimming is
        n*(1.-sum(limits)).  The value of one limit can be set to None to
        indicate an open interval.
    inclusive : {(True,True) tuple}, optional
        Tuple of flags indicating whether the number of data being masked on
        the left (right) end should be truncated (True) or rounded (False) to
        integers.
    axis : {None,int}, optional
        Axis along which to trim. If None, the whole array is trimmed, but its
        shape is maintained.

    """
    def _trimr1D(a, low_limit, up_limit, low_inclusive, up_inclusive):
        n = a.count()
        idx = a.argsort()
        if low_limit:
            if low_inclusive:
                lowidx = int(low_limit*n)
            else:
                lowidx = int(np.round(low_limit*n))
            a[idx[:lowidx]] = masked
        if up_limit is not None:
            if up_inclusive:
                upidx = n - int(n*up_limit)
            else:
                upidx = n - int(np.round(n*up_limit))
            a[idx[upidx:]] = masked
        return a

    a = ma.asarray(a)
    a.unshare_mask()
    if limits is None:
        return a

    # Check the limits
    (lolim, uplim) = limits
    errmsg = "The proportion to cut from the %s should be between 0. and 1."
    if lolim is not None:
        if lolim > 1. or lolim < 0:
            raise ValueError(errmsg % 'beginning' + "(got %s)" % lolim)
    if uplim is not None:
        if uplim > 1. or uplim < 0:
            raise ValueError(errmsg % 'end' + "(got %s)" % uplim)

    (loinc, upinc) = inclusive

    if axis is None:
        shp = a.shape
        return _trimr1D(a.ravel(),lolim,uplim,loinc,upinc).reshape(shp)
    else:
        return ma.apply_along_axis(_trimr1D, axis, a, lolim,uplim,loinc,upinc)


trimdoc = """
    Parameters
    ----------
    a : sequence
        Input array
    limits : {None, tuple}, optional
        If `relative` is False, tuple (lower limit, upper limit) in absolute values.
        Values of the input array lower (greater) than the lower (upper) limit are
        masked.

        If `relative` is True, tuple (lower percentage, upper percentage) to cut
        on each side of the  array, with respect to the number of unmasked data.

        Noting n the number of unmasked data before trimming, the (n*limits[0])th
        smallest data and the (n*limits[1])th largest data are masked, and the
        total number of unmasked data after trimming is n*(1.-sum(limits))
        In each case, the value of one limit can be set to None to indicate an
        open interval.

        If limits is None, no trimming is performed
    inclusive : {(bool, bool) tuple}, optional
        If `relative` is False, tuple indicating whether values exactly equal
        to the absolute limits are allowed.
        If `relative` is True, tuple indicating whether the number of data
        being masked on each side should be rounded (True) or truncated
        (False).
    relative : bool, optional
        Whether to consider the limits as absolute values (False) or proportions
        to cut (True).
    axis : int, optional
        Axis along which to trim.
"""


def trim(a, limits=None, inclusive=(True,True), relative=False, axis=None):
    """
    Trims an array by masking the data outside some given limits.

    Returns a masked version of the input array.

    %s

    Examples
    --------
    >>> from scipy.stats.mstats import trim
    >>> z = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    >>> print(trim(z,(3,8)))
    [-- -- 3 4 5 6 7 8 -- --]
    >>> print(trim(z,(0.1,0.2),relative=True))
    [-- 2 3 4 5 6 7 8 -- --]

    """
    if relative:
        return trimr(a, limits=limits, inclusive=inclusive, axis=axis)
    else:
        return trima(a, limits=limits, inclusive=inclusive)


if trim.__doc__:
    trim.__doc__ = trim.__doc__ % trimdoc


def trimboth(data, proportiontocut=0.2, inclusive=(True,True), axis=None):
    """
    Trims the smallest and largest data values.

    Trims the `data` by masking the ``int(proportiontocut * n)`` smallest and
    ``int(proportiontocut * n)`` largest values of data along the given axis,
    where n is the number of unmasked values before trimming.

    Parameters
    ----------
    data : ndarray
        Data to trim.
    proportiontocut : float, optional
        Percentage of trimming (as a float between 0 and 1).
        If n is the number of unmasked values before trimming, the number of
        values after trimming is ``(1 - 2*proportiontocut) * n``.
        Default is 0.2.
    inclusive : {(bool, bool) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).
    axis : int, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.

    """
    return trimr(data, limits=(proportiontocut,proportiontocut),
                 inclusive=inclusive, axis=axis)


def trimtail(data, proportiontocut=0.2, tail='left', inclusive=(True,True),
             axis=None):
    """
    Trims the data by masking values from one tail.

    Parameters
    ----------
    data : array_like
        Data to trim.
    proportiontocut : float, optional
        Percentage of trimming. If n is the number of unmasked values
        before trimming, the number of values after trimming is
        ``(1 - proportiontocut) * n``.  Default is 0.2.
    tail : {'left','right'}, optional
        If 'left' the `proportiontocut` lowest values will be masked.
        If 'right' the `proportiontocut` highest values will be masked.
        Default is 'left'.
    inclusive : {(bool, bool) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).  Default is
        (True, True).
    axis : int, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.  Default is None.

    Returns
    -------
    trimtail : ndarray
        Returned array of same shape as `data` with masked tail values.

    """
    tail = str(tail).lower()[0]
    if tail == 'l':
        limits = (proportiontocut,None)
    elif tail == 'r':
        limits = (None, proportiontocut)
    else:
        raise TypeError("The tail argument should be in ('left','right')")

    return trimr(data, limits=limits, axis=axis, inclusive=inclusive)


trim1 = trimtail


def trimmed_mean(a, limits=(0.1,0.1), inclusive=(1,1), relative=True,
                 axis=None):
    """Returns the trimmed mean of the data along the given axis.

    %s

    """
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)
    if relative:
        return trimr(a,limits=limits,inclusive=inclusive,axis=axis).mean(axis=axis)
    else:
        return trima(a,limits=limits,inclusive=inclusive).mean(axis=axis)


if trimmed_mean.__doc__:
    trimmed_mean.__doc__ = trimmed_mean.__doc__ % trimdoc


def trimmed_var(a, limits=(0.1,0.1), inclusive=(1,1), relative=True,
                axis=None, ddof=0):
    """Returns the trimmed variance of the data along the given axis.

    %s
    ddof : {0,integer}, optional
        Means Delta Degrees of Freedom. The denominator used during computations
        is (n-ddof). DDOF=0 corresponds to a biased estimate, DDOF=1 to an un-
        biased estimate of the variance.

    """
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)
    if relative:
        out = trimr(a,limits=limits, inclusive=inclusive,axis=axis)
    else:
        out = trima(a,limits=limits,inclusive=inclusive)

    return out.var(axis=axis, ddof=ddof)


if trimmed_var.__doc__:
    trimmed_var.__doc__ = trimmed_var.__doc__ % trimdoc


def trimmed_std(a, limits=(0.1,0.1), inclusive=(1,1), relative=True,
                axis=None, ddof=0):
    """Returns the trimmed standard deviation of the data along the given axis.

    %s
    ddof : {0,integer}, optional
        Means Delta Degrees of Freedom. The denominator used during computations
        is (n-ddof). DDOF=0 corresponds to a biased estimate, DDOF=1 to an un-
        biased estimate of the variance.

    """
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)
    if relative:
        out = trimr(a,limits=limits,inclusive=inclusive,axis=axis)
    else:
        out = trima(a,limits=limits,inclusive=inclusive)
    return out.std(axis=axis,ddof=ddof)


if trimmed_std.__doc__:
    trimmed_std.__doc__ = trimmed_std.__doc__ % trimdoc


def trimmed_stde(a, limits=(0.1,0.1), inclusive=(1,1), axis=None):
    """
    Returns the standard error of the trimmed mean along the given axis.

    Parameters
    ----------
    a : sequence
        Input array
    limits : {(0.1,0.1), tuple of float}, optional
        tuple (lower percentage, upper percentage) to cut  on each side of the
        array, with respect to the number of unmasked data.

        If n is the number of unmasked data before trimming, the values
        smaller than ``n * limits[0]`` and the values larger than
        ``n * `limits[1]`` are masked, and the total number of unmasked
        data after trimming is ``n * (1.-sum(limits))``.  In each case,
        the value of one limit can be set to None to indicate an open interval.
        If `limits` is None, no trimming is performed.
    inclusive : {(bool, bool) tuple} optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).
    axis : int, optional
        Axis along which to trim.

    Returns
    -------
    trimmed_stde : scalar or ndarray

    """
    def _trimmed_stde_1D(a, low_limit, up_limit, low_inclusive, up_inclusive):
        "Returns the standard error of the trimmed mean for a 1D input data."
        n = a.count()
        idx = a.argsort()
        if low_limit:
            if low_inclusive:
                lowidx = int(low_limit*n)
            else:
                lowidx = np.round(low_limit*n)
            a[idx[:lowidx]] = masked
        if up_limit is not None:
            if up_inclusive:
                upidx = n - int(n*up_limit)
            else:
                upidx = n - np.round(n*up_limit)
            a[idx[upidx:]] = masked
        a[idx[:lowidx]] = a[idx[lowidx]]
        a[idx[upidx:]] = a[idx[upidx-1]]
        winstd = a.std(ddof=1)
        return winstd / ((1-low_limit-up_limit)*np.sqrt(len(a)))

    a = ma.array(a, copy=True, subok=True)
    a.unshare_mask()
    if limits is None:
        return a.std(axis=axis,ddof=1)/ma.sqrt(a.count(axis))
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)

    # Check the limits
    (lolim, uplim) = limits
    errmsg = "The proportion to cut from the %s should be between 0. and 1."
    if lolim is not None:
        if lolim > 1. or lolim < 0:
            raise ValueError(errmsg % 'beginning' + "(got %s)" % lolim)
    if uplim is not None:
        if uplim > 1. or uplim < 0:
            raise ValueError(errmsg % 'end' + "(got %s)" % uplim)

    (loinc, upinc) = inclusive
    if (axis is None):
        return _trimmed_stde_1D(a.ravel(),lolim,uplim,loinc,upinc)
    else:
        if a.ndim > 2:
            raise ValueError("Array 'a' must be at most two dimensional, "
                             "but got a.ndim = %d" % a.ndim)
        return ma.apply_along_axis(_trimmed_stde_1D, axis, a,
                                   lolim,uplim,loinc,upinc)


def _mask_to_limits(a, limits, inclusive):
    """Mask an array for values outside of given limits.

    This is primarily a utility function.

    Parameters
    ----------
    a : array
    limits : (float or None, float or None)
    A tuple consisting of the (lower limit, upper limit).  Values in the
    input array less than the lower limit or greater than the upper limit
    will be masked out. None implies no limit.
    inclusive : (bool, bool)
    A tuple consisting of the (lower flag, upper flag).  These flags
    determine whether values exactly equal to lower or upper are allowed.

    Returns
    -------
    A MaskedArray.

    Raises
    ------
    A ValueError if there are no values within the given limits.
    """
    lower_limit, upper_limit = limits
    lower_include, upper_include = inclusive
    am = ma.MaskedArray(a)
    if lower_limit is not None:
        if lower_include:
            am = ma.masked_less(am, lower_limit)
        else:
            am = ma.masked_less_equal(am, lower_limit)

    if upper_limit is not None:
        if upper_include:
            am = ma.masked_greater(am, upper_limit)
        else:
            am = ma.masked_greater_equal(am, upper_limit)

    if am.count() == 0:
        raise ValueError("No array values within given limits")

    return am


def tmean(a, limits=None, inclusive=(True, True), axis=None):
    """
    Compute the trimmed mean.

    Parameters
    ----------
    a : array_like
        Array of values.
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored.  When limits is None (default), then all
        values are used.  Either of the limit values in the tuple can also be
        None representing a half-open interval.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. If None, compute over the
        whole array. Default is None.

    Returns
    -------
    tmean : float

    Notes
    -----
    For more details on `tmean`, see `stats.tmean`.

    Examples
    --------
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 9, 1, 2],
    ...               [8, 7, 8, 2],
    ...               [5, 6, 0, 2],
    ...               [4, 5, 5, 2]])
    ...
    ...
    >>> mstats.tmean(a, (2,5))
    3.3
    >>> mstats.tmean(a, (2,5), axis=0)
    masked_array(data=[4.0, 5.0, 4.0, 2.0],
                 mask=[False, False, False, False],
           fill_value=1e+20)

    """
    return trima(a, limits=limits, inclusive=inclusive).mean(axis=axis)


def tvar(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    """
    Compute the trimmed variance

    This function computes the sample variance of an array of values,
    while ignoring values which are outside of given `limits`.

    Parameters
    ----------
    a : array_like
        Array of values.
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored. When limits is None, then all values are
        used. Either of the limit values in the tuple can also be None
        representing a half-open interval.  The default value is None.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. If None, compute over the
        whole array. Default is zero.
    ddof : int, optional
        Delta degrees of freedom. Default is 1.

    Returns
    -------
    tvar : float
        Trimmed variance.

    Notes
    -----
    For more details on `tvar`, see `stats.tvar`.

    """
    a = a.astype(float).ravel()
    if limits is None:
        n = (~a.mask).sum()  # todo: better way to do that?
        return np.ma.var(a) * n/(n-1.)
    am = _mask_to_limits(a, limits=limits, inclusive=inclusive)

    return np.ma.var(am, axis=axis, ddof=ddof)


def tmin(a, lowerlimit=None, axis=0, inclusive=True):
    """
    Compute the trimmed minimum

    Parameters
    ----------
    a : array_like
        array of values
    lowerlimit : None or float, optional
        Values in the input array less than the given limit will be ignored.
        When lowerlimit is None, then all values are used. The default value
        is None.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    inclusive : {True, False}, optional
        This flag determines whether values exactly equal to the lower limit
        are included.  The default value is True.

    Returns
    -------
    tmin : float, int or ndarray

    Notes
    -----
    For more details on `tmin`, see `stats.tmin`.

    Examples
    --------
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 2, 1, 2],
    ...               [8, 1, 8, 2],
    ...               [5, 3, 0, 2],
    ...               [4, 7, 5, 2]])
    ...
    >>> mstats.tmin(a, 5)
    masked_array(data=[5, 7, 5, --],
                 mask=[False, False, False,  True],
           fill_value=999999)

    """
    a, axis = _chk_asarray(a, axis)
    am = trima(a, (lowerlimit, None), (inclusive, False))
    return ma.minimum.reduce(am, axis)


def tmax(a, upperlimit=None, axis=0, inclusive=True):
    """
    Compute the trimmed maximum

    This function computes the maximum value of an array along a given axis,
    while ignoring values larger than a specified upper limit.

    Parameters
    ----------
    a : array_like
        array of values
    upperlimit : None or float, optional
        Values in the input array greater than the given limit will be ignored.
        When upperlimit is None, then all values are used. The default value
        is None.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    inclusive : {True, False}, optional
        This flag determines whether values exactly equal to the upper limit
        are included.  The default value is True.

    Returns
    -------
    tmax : float, int or ndarray

    Notes
    -----
    For more details on `tmax`, see `stats.tmax`.

    Examples
    --------
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 9, 1, 2],
    ...               [8, 7, 8, 2],
    ...               [5, 6, 0, 2],
    ...               [4, 5, 5, 2]])
    ...
    ...
    >>> mstats.tmax(a, 4)
    masked_array(data=[4, --, 3, 2],
                 mask=[False,  True, False, False],
           fill_value=999999)

    """
    a, axis = _chk_asarray(a, axis)
    am = trima(a, (None, upperlimit), (False, inclusive))
    return ma.maximum.reduce(am, axis)


def tsem(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    """
    Compute the trimmed standard error of the mean.

    This function finds the standard error of the mean for given
    values, ignoring values outside the given `limits`.

    Parameters
    ----------
    a : array_like
        array of values
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored. When limits is None, then all values are
        used. Either of the limit values in the tuple can also be None
        representing a half-open interval.  The default value is None.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. If None, compute over the
        whole array. Default is zero.
    ddof : int, optional
        Delta degrees of freedom. Default is 1.

    Returns
    -------
    tsem : float

    Notes
    -----
    For more details on `tsem`, see `stats.tsem`.

    """
    a = ma.asarray(a).ravel()
    if limits is None:
        n = float(a.count())
        return a.std(axis=axis, ddof=ddof)/ma.sqrt(n)

    am = trima(a.ravel(), limits, inclusive)
    sd = np.sqrt(am.var(axis=axis, ddof=ddof))
    return sd / np.sqrt(am.count())


def winsorize(a, limits=None, inclusive=(True, True), inplace=False,
              axis=None, nan_policy='propagate'):
    """Returns a Winsorized version of the input array.

    The (limits[0])th lowest values are set to the (limits[0])th percentile,
    and the (limits[1])th highest values are set to the (1 - limits[1])th
    percentile.
    Masked values are skipped.


    Parameters
    ----------
    a : sequence
        Input array.
    limits : {None, tuple of float}, optional
        Tuple of the percentages to cut on each side of the array, with respect
        to the number of unmasked data, as floats between 0. and 1.
        Noting n the number of unmasked data before trimming, the
        (n*limits[0])th smallest data and the (n*limits[1])th largest data are
        masked, and the total number of unmasked data after trimming
        is n*(1.-sum(limits)) The value of one limit can be set to None to
        indicate an open interval.
    inclusive : {(True, True) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be truncated (True) or rounded (False).
    inplace : {False, True}, optional
        Whether to winsorize in place (True) or to use a copy (False)
    axis : {None, int}, optional
        Axis along which to trim. If None, the whole array is trimmed, but its
        shape is maintained.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': allows nan values and may overwrite or propagate them
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Notes
    -----
    This function is applied to reduce the effect of possibly spurious outliers
    by limiting the extreme values.

    Examples
    --------
    >>> from scipy.stats.mstats import winsorize

    A shuffled array contains integers from 1 to 10.

    >>> a = np.array([10, 4, 9, 8, 5, 3, 7, 2, 1, 6])

    The 10% of the lowest value (i.e., `1`) and the 20% of the highest
    values (i.e., `9` and `10`) are replaced.

    >>> winsorize(a, limits=[0.1, 0.2])
    masked_array(data=[8, 4, 8, 8, 5, 3, 7, 2, 2, 6],
                 mask=False,
           fill_value=999999)

    """
    def _winsorize1D(a, low_limit, up_limit, low_include, up_include,
                     contains_nan, nan_policy):
        n = a.count()
        idx = a.argsort()
        if contains_nan:
            nan_count = np.count_nonzero(np.isnan(a))
        if low_limit:
            if low_include:
                lowidx = int(low_limit * n)
            else:
                lowidx = np.round(low_limit * n).astype(int)
            if contains_nan and nan_policy == 'omit':
                lowidx = min(lowidx, n-nan_count-1)
            a[idx[:lowidx]] = a[idx[lowidx]]
        if up_limit is not None:
            if up_include:
                upidx = n - int(n * up_limit)
            else:
                upidx = n - np.round(n * up_limit).astype(int)
            if contains_nan and nan_policy == 'omit':
                a[idx[upidx:-nan_count]] = a[idx[upidx - 1]]
            else:
                a[idx[upidx:]] = a[idx[upidx - 1]]
        return a

    contains_nan, nan_policy = scipy.stats.stats._contains_nan(a, nan_policy)
    # We are going to modify a: better make a copy
    a = ma.array(a, copy=np.logical_not(inplace))

    if limits is None:
        return a
    if (not isinstance(limits, tuple)) and isinstance(limits, float):
        limits = (limits, limits)

    # Check the limits
    (lolim, uplim) = limits
    errmsg = "The proportion to cut from the %s should be between 0. and 1."
    if lolim is not None:
        if lolim > 1. or lolim < 0:
            raise ValueError(errmsg % 'beginning' + "(got %s)" % lolim)
    if uplim is not None:
        if uplim > 1. or uplim < 0:
            raise ValueError(errmsg % 'end' + "(got %s)" % uplim)

    (loinc, upinc) = inclusive

    if axis is None:
        shp = a.shape
        return _winsorize1D(a.ravel(), lolim, uplim, loinc, upinc,
                            contains_nan, nan_policy).reshape(shp)
    else:
        return ma.apply_along_axis(_winsorize1D, axis, a, lolim, uplim, loinc,
                                   upinc, contains_nan, nan_policy)


def moment(a, moment=1, axis=0):
    """
    Calculates the nth moment about the mean for a sample.

    Parameters
    ----------
    a : array_like
       data
    moment : int, optional
       order of central moment that is returned
    axis : int or None, optional
       Axis along which the central moment is computed. Default is 0.
       If None, compute over the whole array `a`.

    Returns
    -------
    n-th central moment : ndarray or float
       The appropriate moment along the given axis or over all values if axis
       is None. The denominator for the moment calculation is the number of
       observations, no degrees of freedom correction is done.

    Notes
    -----
    For more details about `moment`, see `stats.moment`.

    """
    a, axis = _chk_asarray(a, axis)
    if a.size == 0:
        moment_shape = list(a.shape)
        del moment_shape[axis]
        dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64
        # empty array, return nan(s) with shape matching `moment`
        out_shape = (moment_shape if np.isscalar(moment)
                    else [len(moment)] + moment_shape)
        if len(out_shape) == 0:
            return dtype(np.nan)
        else:
            return ma.array(np.full(out_shape, np.nan, dtype=dtype))

    # for array_like moment input, return a value for each.
    if not np.isscalar(moment):
        mean = a.mean(axis, keepdims=True)
        mmnt = [_moment(a, i, axis, mean=mean) for i in moment]
        return ma.array(mmnt)
    else:
        return _moment(a, moment, axis)

# Moment with optional pre-computed mean, equal to a.mean(axis, keepdims=True)
def _moment(a, moment, axis, *, mean=None):
    if np.abs(moment - np.round(moment)) > 0:
        raise ValueError("All moment parameters must be integers")

    if moment == 0 or moment == 1:
        # By definition the zeroth moment about the mean is 1, and the first
        # moment is 0.
        shape = list(a.shape)
        del shape[axis]
        dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64

        if len(shape) == 0:
            return dtype(1.0 if moment == 0 else 0.0)
        else:
            return (ma.ones(shape, dtype=dtype) if moment == 0
                    else ma.zeros(shape, dtype=dtype))
    else:
        # Exponentiation by squares: form exponent sequence
        n_list = [moment]
        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n-1)/2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        mean = a.mean(axis, keepdims=True) if mean is None else mean
        a_zero_mean = a - mean
        if n_list[-1] == 1:
            s = a_zero_mean.copy()
        else:
            s = a_zero_mean**2

        # Perform multiplications
        for n in n_list[-2::-1]:
            s = s**2
            if n % 2:
                s *= a_zero_mean
        return s.mean(axis)


def variation(a, axis=0, ddof=0):
    """
    Compute the coefficient of variation.

    The coefficient of variation is the standard deviation divided by the
    mean.  This function is equivalent to::

        np.std(x, axis=axis, ddof=ddof) / np.mean(x)

    The default for ``ddof`` is 0, but many definitions of the coefficient
    of variation use the square root of the unbiased sample variance
    for the sample standard deviation, which corresponds to ``ddof=1``.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate the coefficient of variation. Default
        is 0. If None, compute over the whole array `a`.
    ddof : int, optional
        Delta degrees of freedom.  Default is 0.

    Returns
    -------
    variation : ndarray
        The calculated variation along the requested axis.

    Notes
    -----
    For more details about `variation`, see `stats.variation`.

    Examples
    --------
    >>> from scipy.stats.mstats import variation
    >>> a = np.array([2,8,4])
    >>> variation(a)
    0.5345224838248487
    >>> b = np.array([2,8,3,4])
    >>> c = np.ma.masked_array(b, mask=[0,0,1,0])
    >>> variation(c)
    0.5345224838248487

    In the example above, it can be seen that this works the same as
    `stats.variation` except 'stats.mstats.variation' ignores masked
    array elements.

    """
    a, axis = _chk_asarray(a, axis)
    return a.std(axis, ddof=ddof)/a.mean(axis)


def skew(a, axis=0, bias=True):
    """
    Computes the skewness of a data set.

    Parameters
    ----------
    a : ndarray
        data
    axis : int or None, optional
        Axis along which skewness is calculated. Default is 0.
        If None, compute over the whole array `a`.
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.

    Returns
    -------
    skewness : ndarray
        The skewness of values along an axis, returning 0 where all values are
        equal.

    Notes
    -----
    For more details about `skew`, see `stats.skew`.

    """
    a, axis = _chk_asarray(a,axis)
    mean = a.mean(axis, keepdims=True)
    m2 = _moment(a, 2, axis, mean=mean)
    m3 = _moment(a, 3, axis, mean=mean)
    zero = (m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze(axis))**2)
    with np.errstate(all='ignore'):
        vals = ma.where(zero, 0, m3 / m2**1.5)

    if not bias and zero is not ma.masked and m2 is not ma.masked:
        n = a.count(axis)
        can_correct = ~zero & (n > 2)
        if can_correct.any():
            m2 = np.extract(can_correct, m2)
            m3 = np.extract(can_correct, m3)
            nval = ma.sqrt((n-1.0)*n)/(n-2.0)*m3/m2**1.5
            np.place(vals, can_correct, nval)
    return vals


def kurtosis(a, axis=0, fisher=True, bias=True):
    """
    Computes the kurtosis (Fisher or Pearson) of a dataset.

    Kurtosis is the fourth central moment divided by the square of the
    variance. If Fisher's definition is used, then 3.0 is subtracted from
    the result to give 0.0 for a normal distribution.

    If bias is False then the kurtosis is calculated using k statistics to
    eliminate bias coming from biased moment estimators

    Use `kurtosistest` to see if result is close enough to normal.

    Parameters
    ----------
    a : array
        data for which the kurtosis is calculated
    axis : int or None, optional
        Axis along which the kurtosis is calculated. Default is 0.
        If None, compute over the whole array `a`.
    fisher : bool, optional
        If True, Fisher's definition is used (normal ==> 0.0). If False,
        Pearson's definition is used (normal ==> 3.0).
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.

    Returns
    -------
    kurtosis : array
        The kurtosis of values along an axis. If all values are equal,
        return -3 for Fisher's definition and 0 for Pearson's definition.

    Notes
    -----
    For more details about `kurtosis`, see `stats.kurtosis`.

    """
    a, axis = _chk_asarray(a, axis)
    mean = a.mean(axis, keepdims=True)
    m2 = _moment(a, 2, axis, mean=mean)
    m4 = _moment(a, 4, axis, mean=mean)
    zero = (m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze(axis))**2)
    with np.errstate(all='ignore'):
        vals = ma.where(zero, 0, m4 / m2**2.0)

    if not bias and zero is not ma.masked and m2 is not ma.masked:
        n = a.count(axis)
        can_correct = ~zero & (n > 3)
        if can_correct.any():
            n = np.extract(can_correct, n)
            m2 = np.extract(can_correct, m2)
            m4 = np.extract(can_correct, m4)
            nval = 1.0/(n-2)/(n-3)*((n*n-1.0)*m4/m2**2.0-3*(n-1)**2.0)
            np.place(vals, can_correct, nval+3.0)
    if fisher:
        return vals - 3
    else:
        return vals


DescribeResult = namedtuple('DescribeResult', ('nobs', 'minmax', 'mean',
                                               'variance', 'skewness',
                                               'kurtosis'))


def describe(a, axis=0, ddof=0, bias=True):
    """
    Computes several descriptive statistics of the passed array.

    Parameters
    ----------
    a : array_like
        Data array
    axis : int or None, optional
        Axis along which to calculate statistics. Default 0. If None,
        compute over the whole array `a`.
    ddof : int, optional
        degree of freedom (default 0); note that default ddof is different
        from the same routine in stats.describe
    bias : bool, optional
        If False, then the skewness and kurtosis calculations are corrected for
        statistical bias.

    Returns
    -------
    nobs : int
        (size of the data (discarding missing values)

    minmax : (int, int)
        min, max

    mean : float
        arithmetic mean

    variance : float
        unbiased variance

    skewness : float
        biased skewness

    kurtosis : float
        biased kurtosis

    Examples
    --------
    >>> from scipy.stats.mstats import describe
    >>> ma = np.ma.array(range(6), mask=[0, 0, 0, 1, 1, 1])
    >>> describe(ma)
    DescribeResult(nobs=3, minmax=(masked_array(data=0,
                 mask=False,
           fill_value=999999), masked_array(data=2,
                 mask=False,
           fill_value=999999)), mean=1.0, variance=0.6666666666666666,
           skewness=masked_array(data=0., mask=False, fill_value=1e+20),
            kurtosis=-1.5)

    """
    a, axis = _chk_asarray(a, axis)
    n = a.count(axis)
    mm = (ma.minimum.reduce(a, axis=axis), ma.maximum.reduce(a, axis=axis))
    m = a.mean(axis)
    v = a.var(axis, ddof=ddof)
    sk = skew(a, axis, bias=bias)
    kurt = kurtosis(a, axis, bias=bias)

    return DescribeResult(n, mm, m, v, sk, kurt)


def stde_median(data, axis=None):
    """Returns the McKean-Schrader estimate of the standard error of the sample
    median along the given axis. masked values are discarded.

    Parameters
    ----------
    data : ndarray
        Data to trim.
    axis : {None,int}, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.

    """
    def _stdemed_1D(data):
        data = np.sort(data.compressed())
        n = len(data)
        z = 2.5758293035489004
        k = int(np.round((n+1)/2. - z * np.sqrt(n/4.),0))
        return ((data[n-k] - data[k-1])/(2.*z))

    data = ma.array(data, copy=False, subok=True)
    if (axis is None):
        return _stdemed_1D(data)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        return ma.apply_along_axis(_stdemed_1D, axis, data)


SkewtestResult = namedtuple('SkewtestResult', ('statistic', 'pvalue'))


def skewtest(a, axis=0):
    """
    Tests whether the skew is different from the normal distribution.

    Parameters
    ----------
    a : array
        The data to be tested
    axis : int or None, optional
       Axis along which statistics are calculated. Default is 0.
       If None, compute over the whole array `a`.

    Returns
    -------
    statistic : float
        The computed z-score for this test.
    pvalue : float
        a 2-sided p-value for the hypothesis test

    Notes
    -----
    For more details about `skewtest`, see `stats.skewtest`.

    """
    a, axis = _chk_asarray(a, axis)
    if axis is None:
        a = a.ravel()
        axis = 0
    b2 = skew(a,axis)
    n = a.count(axis)
    if np.min(n) < 8:
        raise ValueError(
            "skewtest is not valid with less than 8 samples; %i samples"
            " were given." % np.min(n))

    y = b2 * ma.sqrt(((n+1)*(n+3)) / (6.0*(n-2)))
    beta2 = (3.0*(n*n+27*n-70)*(n+1)*(n+3)) / ((n-2.0)*(n+5)*(n+7)*(n+9))
    W2 = -1 + ma.sqrt(2*(beta2-1))
    delta = 1/ma.sqrt(0.5*ma.log(W2))
    alpha = ma.sqrt(2.0/(W2-1))
    y = ma.where(y == 0, 1, y)
    Z = delta*ma.log(y/alpha + ma.sqrt((y/alpha)**2+1))

    return SkewtestResult(Z, 2 * distributions.norm.sf(np.abs(Z)))


KurtosistestResult = namedtuple('KurtosistestResult', ('statistic', 'pvalue'))


def kurtosistest(a, axis=0):
    """
    Tests whether a dataset has normal kurtosis

    Parameters
    ----------
    a : array
        array of the sample data
    axis : int or None, optional
       Axis along which to compute test. Default is 0. If None,
       compute over the whole array `a`.

    Returns
    -------
    statistic : float
        The computed z-score for this test.
    pvalue : float
        The 2-sided p-value for the hypothesis test

    Notes
    -----
    For more details about `kurtosistest`, see `stats.kurtosistest`.

    """
    a, axis = _chk_asarray(a, axis)
    n = a.count(axis=axis)
    if np.min(n) < 5:
        raise ValueError(
            "kurtosistest requires at least 5 observations; %i observations"
            " were given." % np.min(n))
    if np.min(n) < 20:
        warnings.warn(
            "kurtosistest only valid for n>=20 ... continuing anyway, n=%i" %
            np.min(n))

    b2 = kurtosis(a, axis, fisher=False)
    E = 3.0*(n-1) / (n+1)
    varb2 = 24.0*n*(n-2.)*(n-3) / ((n+1)*(n+1.)*(n+3)*(n+5))
    x = (b2-E)/ma.sqrt(varb2)
    sqrtbeta1 = 6.0*(n*n-5*n+2)/((n+7)*(n+9)) * np.sqrt((6.0*(n+3)*(n+5)) /
                                                        (n*(n-2)*(n-3)))
    A = 6.0 + 8.0/sqrtbeta1 * (2.0/sqrtbeta1 + np.sqrt(1+4.0/(sqrtbeta1**2)))
    term1 = 1 - 2./(9.0*A)
    denom = 1 + x*ma.sqrt(2/(A-4.0))
    if np.ma.isMaskedArray(denom):
        # For multi-dimensional array input
        denom[denom == 0.0] = masked
    elif denom == 0.0:
        denom = masked

    term2 = np.ma.where(denom > 0, ma.power((1-2.0/A)/denom, 1/3.0),
                        -ma.power(-(1-2.0/A)/denom, 1/3.0))
    Z = (term1 - term2) / np.sqrt(2/(9.0*A))

    return KurtosistestResult(Z, 2 * distributions.norm.sf(np.abs(Z)))


NormaltestResult = namedtuple('NormaltestResult', ('statistic', 'pvalue'))


def normaltest(a, axis=0):
    """
    Tests whether a sample differs from a normal distribution.

    Parameters
    ----------
    a : array_like
        The array containing the data to be tested.
    axis : int or None, optional
        Axis along which to compute test. Default is 0. If None,
        compute over the whole array `a`.

    Returns
    -------
    statistic : float or array
        ``s^2 + k^2``, where ``s`` is the z-score returned by `skewtest` and
        ``k`` is the z-score returned by `kurtosistest`.
    pvalue : float or array
       A 2-sided chi squared probability for the hypothesis test.

    Notes
    -----
    For more details about `normaltest`, see `stats.normaltest`.

    """
    a, axis = _chk_asarray(a, axis)
    s, _ = skewtest(a, axis)
    k, _ = kurtosistest(a, axis)
    k2 = s*s + k*k

    return NormaltestResult(k2, distributions.chi2.sf(k2, 2))


def mquantiles(a, prob=list([.25,.5,.75]), alphap=.4, betap=.4, axis=None,
               limit=()):
    """
    Computes empirical quantiles for a data array.

    Samples quantile are defined by ``Q(p) = (1-gamma)*x[j] + gamma*x[j+1]``,
    where ``x[j]`` is the j-th order statistic, and gamma is a function of
    ``j = floor(n*p + m)``, ``m = alphap + p*(1 - alphap - betap)`` and
    ``g = n*p + m - j``.

    Reinterpreting the above equations to compare to **R** lead to the
    equation: ``p(k) = (k - alphap)/(n + 1 - alphap - betap)``

    Typical values of (alphap,betap) are:
        - (0,1)    : ``p(k) = k/n`` : linear interpolation of cdf
          (**R** type 4)
        - (.5,.5)  : ``p(k) = (k - 1/2.)/n`` : piecewise linear function
          (**R** type 5)
        - (0,0)    : ``p(k) = k/(n+1)`` :
          (**R** type 6)
        - (1,1)    : ``p(k) = (k-1)/(n-1)``: p(k) = mode[F(x[k])].
          (**R** type 7, **R** default)
        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``: Then p(k) ~ median[F(x[k])].
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x.
          (**R** type 8)
        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``: Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed
          (**R** type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM

    Parameters
    ----------
    a : array_like
        Input data, as a sequence or array of dimension at most 2.
    prob : array_like, optional
        List of quantiles to compute.
    alphap : float, optional
        Plotting positions parameter, default is 0.4.
    betap : float, optional
        Plotting positions parameter, default is 0.4.
    axis : int, optional
        Axis along which to perform the trimming.
        If None (default), the input array is first flattened.
    limit : tuple, optional
        Tuple of (lower, upper) values.
        Values of `a` outside this open interval are ignored.

    Returns
    -------
    mquantiles : MaskedArray
        An array containing the calculated quantiles.

    Notes
    -----
    This formulation is very similar to **R** except the calculation of
    ``m`` from ``alphap`` and ``betap``, where in **R** ``m`` is defined
    with each type.

    References
    ----------
    .. [1] *R* statistical software: https://www.r-project.org/
    .. [2] *R* ``quantile`` function:
            http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html

    Examples
    --------
    >>> from scipy.stats.mstats import mquantiles
    >>> a = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])
    >>> mquantiles(a)
    array([ 19.2,  40. ,  42.8])

    Using a 2D array, specifying axis and limit.

    >>> data = np.array([[   6.,    7.,    1.],
    ...                  [  47.,   15.,    2.],
    ...                  [  49.,   36.,    3.],
    ...                  [  15.,   39.,    4.],
    ...                  [  42.,   40., -999.],
    ...                  [  41.,   41., -999.],
    ...                  [   7., -999., -999.],
    ...                  [  39., -999., -999.],
    ...                  [  43., -999., -999.],
    ...                  [  40., -999., -999.],
    ...                  [  36., -999., -999.]])
    >>> print(mquantiles(data, axis=0, limit=(0, 50)))
    [[19.2  14.6   1.45]
     [40.   37.5   2.5 ]
     [42.8  40.05  3.55]]

    >>> data[:, 2] = -999.
    >>> print(mquantiles(data, axis=0, limit=(0, 50)))
    [[19.200000000000003 14.6 --]
     [40.0 37.5 --]
     [42.800000000000004 40.05 --]]

    """
    def _quantiles1D(data,m,p):
        x = np.sort(data.compressed())
        n = len(x)
        if n == 0:
            return ma.array(np.empty(len(p), dtype=float), mask=True)
        elif n == 1:
            return ma.array(np.resize(x, p.shape), mask=nomask)
        aleph = (n*p + m)
        k = np.floor(aleph.clip(1, n-1)).astype(int)
        gamma = (aleph-k).clip(0,1)
        return (1.-gamma)*x[(k-1).tolist()] + gamma*x[k.tolist()]

    data = ma.array(a, copy=False)
    if data.ndim > 2:
        raise TypeError("Array should be 2D at most !")

    if limit:
        condition = (limit[0] < data) & (data < limit[1])
        data[~condition.filled(True)] = masked

    p = np.array(prob, copy=False, ndmin=1)
    m = alphap + p*(1.-alphap-betap)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        return _quantiles1D(data, m, p)

    return ma.apply_along_axis(_quantiles1D, axis, data, m, p)


def scoreatpercentile(data, per, limit=(), alphap=.4, betap=.4):
    """Calculate the score at the given 'per' percentile of the
    sequence a.  For example, the score at per=50 is the median.

    This function is a shortcut to mquantile

    """
    if (per < 0) or (per > 100.):
        raise ValueError("The percentile should be between 0. and 100. !"
                         " (got %s)" % per)

    return mquantiles(data, prob=[per/100.], alphap=alphap, betap=betap,
                      limit=limit, axis=0).squeeze()


def plotting_positions(data, alpha=0.4, beta=0.4):
    """
    Returns plotting positions (or empirical percentile points) for the data.

    Plotting positions are defined as ``(i-alpha)/(n+1-alpha-beta)``, where:
        - i is the rank order statistics
        - n is the number of unmasked values along the given axis
        - `alpha` and `beta` are two parameters.

    Typical values for `alpha` and `beta` are:
        - (0,1)    : ``p(k) = k/n``, linear interpolation of cdf (R, type 4)
        - (.5,.5)  : ``p(k) = (k-1/2.)/n``, piecewise linear function
          (R, type 5)
        - (0,0)    : ``p(k) = k/(n+1)``, Weibull (R type 6)
        - (1,1)    : ``p(k) = (k-1)/(n-1)``, in this case,
          ``p(k) = mode[F(x[k])]``. That's R default (R type 7)
        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``, then
          ``p(k) ~ median[F(x[k])]``.
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x. (R type 8)
        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``, Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed (R type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM
        - (.3175, .3175): used in scipy.stats.probplot

    Parameters
    ----------
    data : array_like
        Input data, as a sequence or array of dimension at most 2.
    alpha : float, optional
        Plotting positions parameter. Default is 0.4.
    beta : float, optional
        Plotting positions parameter. Default is 0.4.

    Returns
    -------
    positions : MaskedArray
        The calculated plotting positions.

    """
    data = ma.array(data, copy=False).reshape(1,-1)
    n = data.count()
    plpos = np.empty(data.size, dtype=float)
    plpos[n:] = 0
    plpos[data.argsort(axis=None)[:n]] = ((np.arange(1, n+1) - alpha) /
                                          (n + 1.0 - alpha - beta))
    return ma.array(plpos, mask=data._mask)


meppf = plotting_positions


def obrientransform(*args):
    """
    Computes a transform on input data (any number of columns).  Used to
    test for homogeneity of variance prior to running one-way stats.  Each
    array in ``*args`` is one level of a factor.  If an `f_oneway()` run on
    the transformed data and found significant, variances are unequal.   From
    Maxwell and Delaney, p.112.

    Returns: transformed data for use in an ANOVA
    """
    data = argstoarray(*args).T
    v = data.var(axis=0,ddof=1)
    m = data.mean(0)
    n = data.count(0).astype(float)
    # result = ((N-1.5)*N*(a-m)**2 - 0.5*v*(n-1))/((n-1)*(n-2))
    data -= m
    data **= 2
    data *= (n-1.5)*n
    data -= 0.5*v*(n-1)
    data /= (n-1.)*(n-2.)
    if not ma.allclose(v,data.mean(0)):
        raise ValueError("Lack of convergence in obrientransform.")

    return data


def sem(a, axis=0, ddof=1):
    """
    Calculates the standard error of the mean of the input array.

    Also sometimes called standard error of measurement.

    Parameters
    ----------
    a : array_like
        An array containing the values for which the standard error is
        returned.
    axis : int or None, optional
        If axis is None, ravel `a` first. If axis is an integer, this will be
        the axis over which to operate. Defaults to 0.
    ddof : int, optional
        Delta degrees-of-freedom. How many degrees of freedom to adjust
        for bias in limited samples relative to the population estimate
        of variance. Defaults to 1.

    Returns
    -------
    s : ndarray or float
        The standard error of the mean in the sample(s), along the input axis.

    Notes
    -----
    The default value for `ddof` changed in scipy 0.15.0 to be consistent with
    `stats.sem` as well as with the most common definition used (like in the R
    documentation).

    Examples
    --------
    Find standard error along the first axis:

    >>> from scipy import stats
    >>> a = np.arange(20).reshape(5,4)
    >>> print(stats.mstats.sem(a))
    [2.8284271247461903 2.8284271247461903 2.8284271247461903
     2.8284271247461903]

    Find standard error across the whole array, using n degrees of freedom:

    >>> print(stats.mstats.sem(a, axis=None, ddof=0))
    1.2893796958227628

    """
    a, axis = _chk_asarray(a, axis)
    n = a.count(axis=axis)
    s = a.std(axis=axis, ddof=ddof) / ma.sqrt(n)
    return s


F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))


def f_oneway(*args):
    """
    Performs a 1-way ANOVA, returning an F-value and probability given
    any number of groups.  From Heiman, pp.394-7.

    Usage: ``f_oneway(*args)``, where ``*args`` is 2 or more arrays,
    one per treatment group.

    Returns
    -------
    statistic : float
        The computed F-value of the test.
    pvalue : float
        The associated p-value from the F-distribution.

    """
    # Construct a single array of arguments: each row is a group
    data = argstoarray(*args)
    ngroups = len(data)
    ntot = data.count()
    sstot = (data**2).sum() - (data.sum())**2/float(ntot)
    ssbg = (data.count(-1) * (data.mean(-1)-data.mean())**2).sum()
    sswg = sstot-ssbg
    dfbg = ngroups-1
    dfwg = ntot - ngroups
    msb = ssbg/float(dfbg)
    msw = sswg/float(dfwg)
    f = msb/msw
    prob = special.fdtrc(dfbg, dfwg, f)  # equivalent to stats.f.sf

    return F_onewayResult(f, prob)


FriedmanchisquareResult = namedtuple('FriedmanchisquareResult',
                                     ('statistic', 'pvalue'))


def friedmanchisquare(*args):
    """Friedman Chi-Square is a non-parametric, one-way within-subjects ANOVA.
    This function calculates the Friedman Chi-square test for repeated measures
    and returns the result, along with the associated probability value.

    Each input is considered a given group. Ideally, the number of treatments
    among each group should be equal. If this is not the case, only the first
    n treatments are taken into account, where n is the number of treatments
    of the smallest group.
    If a group has some missing values, the corresponding treatments are masked
    in the other groups.
    The test statistic is corrected for ties.

    Masked values in one group are propagated to the other groups.

    Returns
    -------
    statistic : float
        the test statistic.
    pvalue : float
        the associated p-value.

    """
    data = argstoarray(*args).astype(float)
    k = len(data)
    if k < 3:
        raise ValueError("Less than 3 groups (%i): " % k +
                         "the Friedman test is NOT appropriate.")

    ranked = ma.masked_values(rankdata(data, axis=0), 0)
    if ranked._mask is not nomask:
        ranked = ma.mask_cols(ranked)
        ranked = ranked.compressed().reshape(k,-1).view(ndarray)
    else:
        ranked = ranked._data
    (k,n) = ranked.shape
    # Ties correction
    repeats = [find_repeats(row) for row in ranked.T]
    ties = np.array([y for x, y in repeats if x.size > 0])
    tie_correction = 1 - (ties**3-ties).sum()/float(n*(k**3-k))

    ssbg = np.sum((ranked.sum(-1) - n*(k+1)/2.)**2)
    chisq = ssbg * 12./(n*k*(k+1)) * 1./tie_correction

    return FriedmanchisquareResult(chisq,
                                   distributions.chi2.sf(chisq, k-1))


BrunnerMunzelResult = namedtuple('BrunnerMunzelResult', ('statistic', 'pvalue'))


def brunnermunzel(x, y, alternative="two-sided", distribution="t"):
    """
    Computes the Brunner-Munzel test on samples x and y

    Missing values in `x` and/or `y` are discarded.

    Parameters
    ----------
    x, y : array_like
        Array of samples, should be one-dimensional.
    alternative :  'less', 'two-sided', or 'greater', optional
        Whether to get the p-value for the one-sided hypothesis ('less'
        or 'greater') or for the two-sided hypothesis ('two-sided').
        Defaults value is 'two-sided' .
    distribution: 't' or 'normal', optional
        Whether to get the p-value by t-distribution or by standard normal
        distribution.
        Defaults value is 't' .

    Returns
    -------
    statistic : float
        The Brunner-Munzer W statistic.
    pvalue : float
        p-value assuming an t distribution. One-sided or
        two-sided, depending on the choice of `alternative` and `distribution`.

    See Also
    --------
    mannwhitneyu : Mann-Whitney rank test on two samples.

    Notes
    -----
    For more details on `brunnermunzel`, see `stats.brunnermunzel`.

    """
    x = ma.asarray(x).compressed().view(ndarray)
    y = ma.asarray(y).compressed().view(ndarray)
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        return BrunnerMunzelResult(np.nan, np.nan)
    rankc = rankdata(np.concatenate((x,y)))
    rankcx = rankc[0:nx]
    rankcy = rankc[nx:nx+ny]
    rankcx_mean = np.mean(rankcx)
    rankcy_mean = np.mean(rankcy)
    rankx = rankdata(x)
    ranky = rankdata(y)
    rankx_mean = np.mean(rankx)
    ranky_mean = np.mean(ranky)

    Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0))
    Sx /= nx - 1
    Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0))
    Sy /= ny - 1

    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)

    if distribution == "t":
        df_numer = np.power(nx * Sx + ny * Sy, 2.0)
        df_denom = np.power(nx * Sx, 2.0) / (nx - 1)
        df_denom += np.power(ny * Sy, 2.0) / (ny - 1)
        df = df_numer / df_denom
        p = distributions.t.cdf(wbfn, df)
    elif distribution == "normal":
        p = distributions.norm.cdf(wbfn)
    else:
        raise ValueError(
            "distribution should be 't' or 'normal'")

    if alternative == "greater":
        pass
    elif alternative == "less":
        p = 1 - p
    elif alternative == "two-sided":
        p = 2 * np.min([p, 1-p])
    else:
        raise ValueError(
            "alternative should be 'less', 'greater' or 'two-sided'")

    return BrunnerMunzelResult(wbfn, p)