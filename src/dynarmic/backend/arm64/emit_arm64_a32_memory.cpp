/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/ir/acc_type.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

static bool IsOrdered(IR::AccType acctype) {
    return acctype == IR::AccType::ORDERED || acctype == IR::AccType::ORDEREDRW || acctype == IR::AccType::LIMITEDORDERED;
}

template<std::size_t bit_size>
void EmitInlineReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Waddr = ctx.reg_alloc.ReadW(args[1]);
    auto to = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bit_size, 32)>(inst);
    RegAlloc::Realize(Waddr, to);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    code.MOV(Xscratch0, ctx.conf.fastmem_addr);
    if constexpr (bit_size == 8)
        code.LDRB(to, Xscratch0, Waddr, oaknut::IndexExt::UXTW);
    else if constexpr (bit_size == 16)
        code.LDRH(to, Xscratch0, Waddr, oaknut::IndexExt::UXTW);
    else if constexpr (bit_size == 32 || bit_size == 64)
        code.LDR(to, Xscratch0, Waddr, oaknut::IndexExt::UXTW);
    else
        static_assert(bit_size == 8 || bit_size == 16 || bit_size == 32 || bit_size == 64);

    if (ordered)
        code.DMB(oaknut::BarrierOp::ISH);
}

template<std::size_t bit_size>
void EmitInlineExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Waddr = ctx.reg_alloc.ReadW(args[1]);
    auto to = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bit_size, 32)>(inst);
    RegAlloc::Realize(Waddr, to);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    code.MOV(Xscratch0, ctx.conf.fastmem_addr);
    code.ADD(Xscratch0, Xscratch0, Waddr, oaknut::AddSubExt::UXTW);
    if constexpr (bit_size == 8)
        code.LDXRB(to, Xscratch0);
    else if constexpr (bit_size == 16)
        code.LDXRH(to, Xscratch0);
    else if constexpr (bit_size == 32 || bit_size == 64)
        code.LDXR(to, Xscratch0);
    else
        static_assert(bit_size == 8 || bit_size == 16 || bit_size == 32 || bit_size == 64);

    if (ordered)
        code.DMB(oaknut::BarrierOp::ISH);
}

template<std::size_t bit_size>
void EmitInlineWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Waddr = ctx.reg_alloc.ReadW(args[1]);
    auto value = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bit_size, 32)>(args[2]);
    RegAlloc::Realize(Waddr, value);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    code.MOV(Xscratch0, ctx.conf.fastmem_addr);
    if (ordered)
        code.DMB(oaknut::BarrierOp::ISH);

    if constexpr (bit_size == 8)
        code.STRB(value, Xscratch0, Waddr, oaknut::IndexExt::UXTW);
    else if constexpr (bit_size == 16)
        code.STRH(value, Xscratch0, Waddr, oaknut::IndexExt::UXTW);
    else if constexpr (bit_size == 32 || bit_size == 64)
        code.STR(value, Xscratch0, Waddr, oaknut::IndexExt::UXTW);
    else
        static_assert(bit_size == 8 || bit_size == 16 || bit_size == 32 || bit_size == 64);

    if (ordered)
        code.DMB(oaknut::BarrierOp::ISH);
}

template<std::size_t bit_size>
void EmitInlineExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Waddr = ctx.reg_alloc.ReadW(args[1]);
    auto value = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bit_size, 32)>(args[2]);
    RegAlloc::Realize(Wresult, Waddr, value);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    code.MOV(Xscratch0, ctx.conf.fastmem_addr);
    code.ADD(Xscratch0, Xscratch0, Waddr, oaknut::AddSubExt::UXTW);
    if (ordered)
        code.DMB(oaknut::BarrierOp::ISH);

    if constexpr (bit_size == 8)
        code.STXRB(Wresult, value, Xscratch0);
    else if constexpr (bit_size == 16)
        code.STXRH(Wresult, value, Xscratch0);
    else if constexpr (bit_size == 32 || bit_size == 64)
        code.STXR(Wresult, value, Xscratch0);
    else
        static_assert(bit_size == 8 || bit_size == 16 || bit_size == 32 || bit_size == 64);

    if (ordered)
        code.DMB(oaknut::BarrierOp::ISH);
}

static void EmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, LinkTarget fn) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall(inst, {}, args[1]);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    EmitRelocation(code, ctx, fn);
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
}

static void EmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, LinkTarget fn) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall(inst, {}, args[1]);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    code.MOV(Wscratch0, 1);
    code.STRB(Wscratch0, Xstate, offsetof(A32JitState, exclusive_state));
    EmitRelocation(code, ctx, fn);
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
}

static void EmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, LinkTarget fn) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall(inst, {}, args[1], args[2]);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
    EmitRelocation(code, ctx, fn);
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
}

static void EmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, LinkTarget fn) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall(inst, {}, args[1], args[2]);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    oaknut::Label end;

    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
    code.LDRB(Wscratch0, Xstate, offsetof(A32JitState, exclusive_state));
    code.CBZ(Wscratch0, end);
    code.STRB(WZR, Xstate, offsetof(A32JitState, exclusive_state));
    EmitRelocation(code, ctx, fn);
    if (ordered) {
        code.DMB(oaknut::BarrierOp::ISH);
    }
    code.l(end);
}

template<>
void EmitIR<IR::Opcode::A32ClearExclusive>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst*) {
    if (ctx.conf.enable_fastmem)
        code.CLREX();
    else
        code.STR(WZR, Xstate, offsetof(A32JitState, exclusive_state));
}

template<>
void EmitIR<IR::Opcode::A32ReadMemory8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineReadMemory<8>(code, ctx, inst);
    else
        EmitReadMemory(code, ctx, inst, LinkTarget::ReadMemory8);
}

template<>
void EmitIR<IR::Opcode::A32ReadMemory16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineReadMemory<16>(code, ctx, inst);
    else
        EmitReadMemory(code, ctx, inst, LinkTarget::ReadMemory16);
}

template<>
void EmitIR<IR::Opcode::A32ReadMemory32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineReadMemory<32>(code, ctx, inst);
    else
        EmitReadMemory(code, ctx, inst, LinkTarget::ReadMemory32);
}

template<>
void EmitIR<IR::Opcode::A32ReadMemory64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineReadMemory<64>(code, ctx, inst);
    else
        EmitReadMemory(code, ctx, inst, LinkTarget::ReadMemory64);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveReadMemory8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineExclusiveReadMemory<8>(code, ctx, inst);
    else
        EmitExclusiveReadMemory(code, ctx, inst, LinkTarget::ExclusiveReadMemory8);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveReadMemory16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineExclusiveReadMemory<16>(code, ctx, inst);
    else
        EmitExclusiveReadMemory(code, ctx, inst, LinkTarget::ExclusiveReadMemory16);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveReadMemory32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineExclusiveReadMemory<32>(code, ctx, inst);
    else
        EmitExclusiveReadMemory(code, ctx, inst, LinkTarget::ExclusiveReadMemory32);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveReadMemory64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineExclusiveReadMemory<64>(code, ctx, inst);
    else
        EmitExclusiveReadMemory(code, ctx, inst, LinkTarget::ExclusiveReadMemory64);
}

template<>
void EmitIR<IR::Opcode::A32WriteMemory8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineWriteMemory<8>(code, ctx, inst);
    else
        EmitWriteMemory(code, ctx, inst, LinkTarget::WriteMemory8);
}

template<>
void EmitIR<IR::Opcode::A32WriteMemory16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineWriteMemory<16>(code, ctx, inst);
    else
        EmitWriteMemory(code, ctx, inst, LinkTarget::WriteMemory16);
}

template<>
void EmitIR<IR::Opcode::A32WriteMemory32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineWriteMemory<32>(code, ctx, inst);
    else
        EmitWriteMemory(code, ctx, inst, LinkTarget::WriteMemory32);
}

template<>
void EmitIR<IR::Opcode::A32WriteMemory64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineWriteMemory<64>(code, ctx, inst);
    else
        EmitWriteMemory(code, ctx, inst, LinkTarget::WriteMemory64);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveWriteMemory8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineExclusiveWriteMemory<8>(code, ctx, inst);
    else
        EmitExclusiveWriteMemory(code, ctx, inst, LinkTarget::ExclusiveWriteMemory8);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveWriteMemory16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineExclusiveWriteMemory<16>(code, ctx, inst);
    else
        EmitExclusiveWriteMemory(code, ctx, inst, LinkTarget::ExclusiveWriteMemory16);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveWriteMemory32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineExclusiveWriteMemory<32>(code, ctx, inst);
    else
        EmitExclusiveWriteMemory(code, ctx, inst, LinkTarget::ExclusiveWriteMemory32);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveWriteMemory64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (ctx.conf.enable_fastmem)
        EmitInlineExclusiveWriteMemory<64>(code, ctx, inst);
    else
        EmitExclusiveWriteMemory(code, ctx, inst, LinkTarget::ExclusiveWriteMemory64);
}

}  // namespace Dynarmic::Backend::Arm64
