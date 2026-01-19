#include "MetadataStructs.hpp"

//#########################//
// FREE OPERATOR OVERLOADS //
//#########################//

#define DEFINE_DIMENSIONS_BINOP_OVERLOAD(op)                              \
Dimensions operator op (const Dimensions& dims1, const Dimensions& dims2) \
{                                                                         \
    return Dimensions                                                     \
    {                                                                     \
        .Ls = dims1.Ls op dims2.Ls,                                       \
        .x  = dims1.x  op dims2.x ,                                       \
        .y  = dims1.y  op dims2.y ,                                       \
        .z  = dims1.z  op dims2.z ,                                       \
        .t  = dims1.t  op dims2.t                                         \
    };                                                                    \
}

DEFINE_DIMENSIONS_BINOP_OVERLOAD(+)
DEFINE_DIMENSIONS_BINOP_OVERLOAD(-)
DEFINE_DIMENSIONS_BINOP_OVERLOAD(*)
DEFINE_DIMENSIONS_BINOP_OVERLOAD(/)
DEFINE_DIMENSIONS_BINOP_OVERLOAD(%)
DEFINE_DIMENSIONS_BINOP_OVERLOAD(&&)
DEFINE_DIMENSIONS_BINOP_OVERLOAD(||)
DEFINE_DIMENSIONS_BINOP_OVERLOAD(>)
DEFINE_DIMENSIONS_BINOP_OVERLOAD(>=)
DEFINE_DIMENSIONS_BINOP_OVERLOAD(<)
DEFINE_DIMENSIONS_BINOP_OVERLOAD(<=)
#undef DEFINE_DIMENSIONS_OP_OVERLOAD

std::ostream& operator<<(std::ostream& os, const Dimensions& dims)
{
    os  << (dims.Ls) << ", " 
        << (dims.x)  << ", " 
        << (dims.y)  << ", " 
        << (dims.z)  << ", " 
        << (dims.t);
    return os;
}

bool operator==(const Dimensions& dims1, const Dimensions& dims2)
{
    return dims1.Ls == dims2.Ls
        && dims1.x  == dims2.x
        && dims1.y  == dims2.y
        && dims1.z  == dims2.z
        && dims1.t  == dims2.t;
}

Count_t flattenDimensionsLsSlow(const Dimensions& position, const Dimensions& boundary)
{
    Count_t idx = 0;
    idx += position.Ls;
    idx *= boundary.t;
    idx += position.t;
    idx *= boundary.z;
    idx += position.z;
    idx *= boundary.y;
    idx += position.y;
    idx *= boundary.x;
    idx += position.x;

    return idx;
}

Count_t flattenDimensionsLsFast(const Dimensions& position, const Dimensions& boundary)
{
    Count_t idx = 0;
    idx += position.t;
    idx *= boundary.z;
    idx += position.z;
    idx *= boundary.y;
    idx += position.y;
    idx *= boundary.x;
    idx += position.x;
    idx *= boundary.Ls;
    idx += position.Ls;

    return idx;
}

std::ostream& operator<<(std::ostream& os, const FileLocationInfo& loc)
{
    os  << "Dir: " << loc.directory << " File: " << loc.file;
    return os;
}

std::ostream& operator<<(std::ostream& os, const BasisDataLocationInfo& loc)
{
    os  << "Block: " << loc.block_idx << " BV: " << loc.basis_vector_idx << " " << "Site: " << loc.block_site_idx;
    return os;
}

std::ostream& operator<<(std::ostream& os, const CoarseVectorDataLocationInfo& loc)
{
    os  << "Block: " << loc.block_idx << " EV: " << loc.eigenvector_idx;
    return os;
}